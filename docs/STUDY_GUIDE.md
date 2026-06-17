# Guia de Estudo — defendendo este projeto numa entrevista

Este guia tem duas partes:

1. **Conceitos-chave** que o projeto exercita, explicados na profundidade que uma
   entrevista cobra.
2. **Perguntas prováveis** (da mais básica à mais capciosa) com respostas que você
   pode adaptar com suas próprias palavras.

A regra de ouro: numa entrevista, **não recite — explique o porquê e o trade-off**.
Quem entende o trade-off entende o conceito.

---

## Parte 1 — Conceitos-chave

### 1. Por que feature stores existem

O problema central é a **distância entre treino e produção**. O modelo é treinado
em batch, sobre meses de histórico; em produção, ele precisa das mesmas features
calculadas em tempo real, para uma única transação, em milissegundos. Sem uma
camada comum, essas duas implementações divergem e nasce o **training/serving
skew** — o modelo vê em produção números diferentes dos do treino.

Uma feature store resolve quatro dores de uma vez:

- **Consistência treino/serving** — a mesma definição de feature serve os dois lados.
- **Reuso** — uma feature definida uma vez é usada por vários modelos/times.
- **Point-in-time correctness** — montar dados de treino sem vazar o futuro.
- **Serving de baixa latência** — ler o último valor da feature rápido o suficiente
  para decidir durante a transação.

> Frase de efeito: *"feature store é menos sobre guardar números e mais sobre
> garantir que o número que o modelo viu no treino é o mesmo que ele vê em
> produção — no tempo certo."*

### 2. Offline store vs. online store

São o **mesmo dado, otimizado para dois acessos opostos**:

| | Offline store | Online store |
|---|---|---|
| Pergunta que responde | "Quais eram as features de TODAS as contas ao longo do tempo?" | "Quais são as features ATUAIS da conta X?" |
| Padrão de acesso | varredura de alto volume | lookup pontual por chave |
| Latência | segundos/minutos (tudo bem) | milissegundos (crítico) |
| Volume | meses/anos de histórico | um valor por entidade |
| Tecnologia típica | Parquet, BigQuery, Snowflake, Postgres | Redis, DynamoDB, Cassandra |
| Usado por | **treino** e scoring em batch | **inferência online** |
| Neste projeto | Parquet (`FileSource`) | Redis |

A ponte entre os dois é a **materialização** (ver item 4).

### 3. Point-in-time correctness (o conceito mais importante)

Ao montar o dataset de treino, para cada evento rotulado no instante `t` você
precisa das features **como elas eram em `t`**, não as de hoje. Usar o valor atual
é **data leakage**: você ensina o modelo com informação que não existia no momento
da decisão. O sintoma clássico: métricas lindas no treino, desempenho ruim em
produção.

Exemplo concreto deste repo: a feature `amount_avg` (ticket médio do cliente).
Para a transação de 10/jan, ela deve refletir só as compras **até 9/jan**. Se eu
calcular o ticket médio sobre o histórico inteiro (incluindo compras de fevereiro),
estou usando o futuro.

Como o Feast faz: o `get_historical_features` recebe um *entity dataframe* com
`(entidade, event_timestamp)` por linha e faz um **as-of join** — pega o valor da
feature cujo timestamp é o maior que ainda seja `<=` ao do evento (e dentro do
TTL). No código, a garantia começa antes, em `compute_features.py`: o `.shift(1)`
no `expanding().mean()` faz cada linha enxergar só o passado.

> Por que é difícil "na mão": um `>=` no lugar de `<`, esquecer o TTL, ou juntar
> pelo timestamp de ingestão em vez do de evento — qualquer um vaza o futuro. A
> feature store encapsula esse join corretamente.

### 4. Materialização

É o processo de **copiar os valores mais recentes** de cada feature da offline
store para a online store. É o passo que muita gente esquece: o Redis não se
popula sozinho.

- `materialize(start, end)` / `materialize_incremental(end)` varrem a janela na
  offline store e gravam, para cada entidade, o **último** valor no online store.
- Em produção roda **agendado** (a cada hora/dia, via Airflow/cron) ou por
  **streaming** (push de eventos). A frequência define a **frescura** (*freshness*)
  das features online — um trade-off entre custo e o quão atualizado o modelo está.

### 5. Feature view, entity, data source

- **Entity** — a chave de negócio das features. Aqui é a `account` (`account_id`).
  Toda feature é uma propriedade de uma entidade num instante.
- **Data source** — onde os dados físicos vivem e qual coluna é o `event_timestamp`
  (hora do evento) e o `created` (hora de ingestão). Aqui um `FileSource` Parquet.
- **Feature view** — agrupa features que vêm da mesma fonte, compartilham entidade
  e TTL. É a unidade que se materializa e que se pede no treino. Aqui,
  `customer_profile`.
- **Registry** — catálogo versionado de tudo isso. `feast apply` escreve nele.

### 6. On-demand feature view (transformações em tempo de request)

Algumas features só podem ser calculadas **quando a transação chega**, porque
dependem dos dados dela (valor, local, hora). Não dá para pré-computar o "z-score
do valor desta compra" antes de a compra existir.

A `transaction_risk` deste repo combina o perfil armazenado (`amount_avg`,
`amount_std`, `last_txn_*`) com o request (`amount`, `lat`, `lon`, `txn_unixtime`)
para produzir `amount_zscore`, `distance_from_last_km` e `velocity_kmh`. O ponto
de ouro: ela usa **as mesmas funções** (`fraud_fs.transforms`) que o cálculo
offline — então é impossível haver skew entre treino e serving para essas features.

### 7. TTL (time-to-live) das features

Quanto tempo um valor continua "válido" para o point-in-time join. Se o TTL é 90
dias e o perfil mais recente de uma conta tem 100 dias, o Feast **não** o usa
(considera obsoleto) e retorna nulo. Protege contra servir features velhas demais.

### 8. Training/serving skew (e como este projeto o elimina)

Skew acontece quando a feature no treino ≠ a feature no serving. Fontes comuns:
fórmulas diferentes, janelas diferentes, fusos, ou dados online desatualizados.
Defesas neste projeto: (a) transformações puras compartilhadas; (b) a mesma lista
`FEATURE_REFS` no treino e na inferência; (c) a mesma on-demand FV nos dois lados.

### 9. Classe desbalanceada (o lado ML)

Fraude é rara (~1,5%). Implicações que você precisa saber defender:

- **Acurácia é inútil** — "nunca é fraude" acerta 98,5%.
- Use **PR-AUC** (average precision) e **precision/recall**, não só ROC-AUC. PR-AUC
  é mais sensível ao desempenho na classe rara.
- **`class_weight="balanced"`** ou reamostragem (SMOTE/undersampling) para o modelo
  não ignorar a minoria.
- O **threshold** de decisão é de negócio, não 0.5: depende do custo relativo de um
  falso positivo (bloquear cliente legítimo) vs. falso negativo (deixar passar
  fraude).

### 10. Split temporal vs. aleatório

Em dados temporais, treine no passado e teste no futuro. Um `train_test_split`
aleatório coloca eventos futuros no treino e infla as métricas (uma forma de
leakage). Aqui ordenamos por `event_timestamp` e cortamos 80/20 no tempo.

---

## Parte 2 — Perguntas prováveis de entrevista (com respostas)

**P: O que é uma feature store e que problema ela resolve?**
R: É a camada que centraliza definição, armazenamento e serving de features para
ML. Resolve training/serving skew, reuso entre times, point-in-time correctness e
serving de baixa latência. Sem ela, a mesma feature é reimplementada em batch (para
treino) e em tempo real (para produção), e as duas divergem.

**P: Diferença entre offline e online store?**
R: Mesmo dado, acessos opostos. Offline guarda o histórico completo para treino e é
otimizada para varredura de alto volume (aqui Parquet). Online guarda o último
valor por entidade para inferência e é otimizada para lookup em milissegundos (aqui
Redis). A materialização leva da offline para a online.

**P: O que é point-in-time correctness e por que importa?**
R: É montar o dataset de treino com as features como elas eram no instante de cada
evento, não as atuais. Importa porque usar valores atuais vaza o futuro (data
leakage), dando métricas otimistas que somem em produção. O Feast faz isso com um
as-of join (feature.timestamp `<=` evento.timestamp, dentro do TTL); no meu código
o `.shift(1)` garante que cada feature só enxerga o passado.

**P: Mostre, no seu projeto, onde o leakage seria evitado.**
R: Em dois lugares. No `compute_features.py`, `expanding().mean().shift(1)` faz a
média histórica excluir a transação atual; e nas contagens, o `- 1` remove a linha
corrente da janela. No `offline.py`, o `get_historical_features` faz o as-of join
por `event_timestamp`. Tenho um teste (`test_no_leakage_average_uses_only_past`)
que prova que a média na linha `i` é a média de `[0:i]`.

**P: O que é materialização? Quando roda?**
R: É copiar o valor mais recente de cada feature da offline para a online store.
Roda agendada (cron/Airflow) ou via streaming. A frequência define a frescura das
features online — trade-off entre custo de computação e o quão atual o modelo está.

**P: O que é uma on-demand feature view? Por que usar?**
R: Uma feature calculada no momento do request, porque depende dos dados da própria
transação (valor, local, hora) — não dá para pré-computar. Uso para z-score do
valor, distância e velocidade. A vantagem é zero skew: a mesma função roda no treino
e no serving. O custo é latência por request e, no Feast, ODFV é experimental para
retrieval offline em larga escala.

**P: Por que Parquet na offline e Redis na online? O que mudaria em produção?**
R: Parquet é simples, colunar e reprodutível — ótimo para portfólio, mas não escala
para concorrência nem governança. Redis dá leitura sub-milissegundo, ao custo de
memória e necessidade de HA. Em produção eu usaria BigQuery/Snowflake (offline) e
manteria Redis ou DynamoDB (online); no Feast isso é uma troca no
`feature_store.yaml`, sem mexer no resto.

**P: O Feast calcula features para você?**
R: No OSS, não (a engine de transformação batch é limitada). Ele armazena,
versiona e serve. A engenharia de feature é um pipeline a montante — no meu caso
`compute_features.py`. Isso me dá controle e testabilidade, mas me torna
responsável pela orquestração e pela re-materialização.

**P: Como você avalia o modelo, sendo fraude tão rara?**
R: Não uso acurácia. Uso PR-AUC e precision/recall, com split temporal (treino no
passado, teste no futuro). Uso `class_weight="balanced"` e escolho o threshold pelo
custo de negócio de falso positivo vs. falso negativo, não 0.5.

**P: Quais features mais pesaram e isso faz sentido?**
R: `amount_to_avg_ratio`, `velocity_kmh`, `seconds_since_last` e `amount_zscore` —
exatamente os sinais que injetei (valor fora do perfil e viagem impossível). Isso é
uma checagem de sanidade: se a importância não batesse com os padrões plantados, eu
teria um bug na pipeline de features.

**P: Como você garante que treino e serving usam a mesma feature?**
R: Três defesas: transformações puras compartilhadas (`transforms.py`), uma única
lista de referências (`FEATURE_REFS` em `config.py`) usada nos dois caminhos, e a
mesma on-demand FV aplicada tanto no `get_historical_features` quanto no
`get_online_features`.

**P: O que acontece com um cliente novo (cold start)?**
R: Ele não tem perfil materializado, então o online store retorna nulos. O
`SimpleImputer` no Pipeline preenche com a mediana, e o modelo ainda pontua usando
as on-demand features que dependem só do request. É uma decisão explícita — em
produção eu poderia ter regras específicas para contas sem histórico.

**P: Como você escalaria isso?**
R: Offline para warehouse; materialização agendada/streaming; computação de
features em Spark/Dask ou via streaming (Flink/Kafka) para janelas em tempo real;
registry em Postgres compartilhado; e serving atrás de um autoscaler com SLO de
latência. O Feast já abstrai a troca de stores.

**P: Qual a maior fraqueza do projeto?**
R: Dados sintéticos — não têm a sutileza de fraude real, então as métricas são
otimistas. E a materialização é um one-shot, sem o componente de streaming que a
detecção de fraude real exige para frescura de segundos. Eu deixaria isso explícito
e mostraria onde plugar Kafka/Flink.
