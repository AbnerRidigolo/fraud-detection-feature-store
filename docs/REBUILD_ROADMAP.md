# Roteiro de Reconstrução Mental

Ler código não é o mesmo que dominá-lo. Este roteiro te leva a **reconstruir cada
parte do zero**, na ordem em que ela foi construída, com a pergunta que cada etapa
responde e um teste de "você realmente entendeu". Faça de caderno fechado: tente
explicar/escrever antes de abrir o arquivo.

A ordem segue os commits do projeto: **dados → features → offline → online/serving
→ modelo → infra**. Cada peça depende da anterior; reconstruir nessa ordem é o que
faz o modelo mental "encaixar".

---

## Etapa 0 — O mapa de uma frase

Antes de qualquer código, saiba desenhar isto numa folha:

> Transações → perfil point-in-time (offline/Parquet) → **registry** define tudo →
> materializa para Redis (online) → treino lê histórico com as-of join, serving lê
> último valor — **as mesmas features** nos dois lados.

Se você consegue desenhar esse diagrama e dizer o papel de cada caixa, o resto é
detalhe de implementação. Volte a este desenho sempre que se perder.

---

## Etapa 1 — Geração de dados (`generate_data.py`)

**Pergunta que responde:** como ter um problema de fraude com verdade de base
conhecida e reprodutível?

**Reconstrua mentalmente:**
1. Modele o cliente, não só a transação. Cada conta tem **sede geográfica** e
   **perfil de gasto** (média, desvio). Por quê? Porque fraude é desvio do *normal
   daquele cliente* — sem perfil, não há "anormal".
2. Gere o tráfego legítimo perto de casa, dentro do perfil.
3. Injete três fraudes que mapeiam para três features depois:
   - valor fora do perfil → vira `amount_zscore`;
   - sequência rápida → vira `txn_count_1h`;
   - viagem impossível → vira `velocity_kmh`.
4. Semeie tudo (`SEED`) para reprodutibilidade.

**Teste de entendimento:** "Por que injetar a fraude você mesmo em vez de baixar um
dataset?" — Para poder *provar* que cada feature captura um sinal específico (e
checar isso na importância das features no fim).

---

## Etapa 2 — Features point-in-time (`compute_features.py` + `transforms.py`)

**Pergunta que responde:** como calcular o perfil de cada cliente sem vazar o
futuro?

**Reconstrua mentalmente:**
1. Ordene as transações de cada conta por tempo.
2. Para cada transação, o perfil deve refletir **só o que veio antes**. A mecânica:
   `expanding().mean().shift(1)` (o `shift` joga a estatística acumulada para a
   linha seguinte) e, nas contagens em janela, `rolling('1h').count() - 1` (o `-1`
   remove a própria linha).
3. Guarde o estado da transação anterior (`last_txn_lat/lon/unixtime`) — é o que
   permite calcular distância/velocidade depois, no request.
4. Separe a matemática pura em `transforms.py`. Por quê? Para reusar **idêntica** no
   online (anti-skew) e testar isolada.

**Teste de entendimento:** escreva, sem rodar, o valor de `amount_avg` para a 3ª
transação de uma conta com valores [100, 200, 300]. Resposta: 150 (média de
[100,200], não inclui 300). Se você errou incluindo o 300, você não internalizou o
point-in-time ainda.

---

## Etapa 3 — Offline store + definições + treino-dataset (`definitions.py`, `pipeline.py apply`, `offline.py`)

**Pergunta que responde:** como transformar Parquet num dataset de treino
point-in-time-correct?

**Reconstrua mentalmente:**
1. Declare o **contrato**: entidade (`account`), fonte (`FileSource` com
   `event_timestamp` e `created`), feature view (`customer_profile`, com TTL) e a
   **request source** + **on-demand FV** (`transaction_risk`).
2. `feast apply` registra isso no **registry**. Nada existe para o Feast antes disso.
3. Monte o **entity dataframe**: uma linha por evento rotulado, com
   `(account_id, event_timestamp, label)` + os campos do request (para a ODFV).
4. `get_historical_features` faz o **as-of join**: para cada linha, pega o perfil
   cujo timestamp é o maior `<=` ao do evento (dentro do TTL) e calcula as
   on-demand features em cima.

**Teste de entendimento:** "Se eu pedir as features para um evento de 10/jan, e o
perfil mais recente da conta for de 8/jan e outro de 12/jan, qual o Feast usa?" — O
de 8/jan (o maior que ainda é `<=` 10/jan). Usar o de 12/jan seria vazar o futuro.

---

## Etapa 4 — Online store + serving (`pipeline.py materialize`, `inference.py`, `serve.py`)

**Pergunta que responde:** como servir as mesmas features em milissegundos para uma
transação que está chegando?

**Reconstrua mentalmente:**
1. **Materialize**: copie o último valor de cada conta da offline para o Redis. Sem
   isso, o online store está vazio — erro nº 1 de iniciante.
2. Na inferência, monte o `entity_row` com a chave (`account_id`) **e** os dados do
   request (`amount`, `lat`, `lon`, `txn_unixtime`).
3. `get_online_features` busca o perfil no Redis e calcula a ODFV no request.
4. Monte o vetor na **mesma ordem** (`MODEL_FEATURES`) do treino e pontue.

**Teste de entendimento:** "Por que `get_online_features` precisa dos dados do
request, se ele lê do Redis?" — Porque a on-demand FV (z-score, velocidade) depende
da transação atual; o Redis só tem o perfil pré-computado.

---

## Etapa 5 — Modelo (`train.py`)

**Pergunta que responde:** como treinar e avaliar honestamente com fraude rara?

**Reconstrua mentalmente:**
1. **Split temporal** (não aleatório): passado treina, futuro testa.
2. Pipeline = `SimpleImputer` (cold start) + `RandomForest(class_weight="balanced")`.
3. Avalie com **PR-AUC + precision/recall**, nunca acurácia.
4. Olhe `feature_importances_` e confronte com os padrões injetados — é a sua
   checagem de sanidade da pipeline inteira.

**Teste de entendimento:** "Seu modelo tem 99% de acurácia. Isso é bom?" — Sozinho,
não diz nada: prever 'nunca fraude' já dá ~98,5%. O que importa é
recall/precision/PR-AUC na classe fraude.

---

## Etapa 6 — Infra (Docker, Compose, CI)

**Pergunta que responde:** como tornar isso reproduzível e provado por terceiros?

**Reconstrua mentalmente:**
1. Uma imagem, dois comandos (pipeline one-shot e API) — menos drift.
2. Compose: `redis` (online store) + `pipeline` (job que sai) + `api` (sobe após o
   pipeline). Volume `./data` compartilha registry e modelo.
3. CI em dois jobs: lint+unit (rápido) e **e2e com Redis de verdade** (prova o ciclo
   completo em escala pequena).

**Teste de entendimento:** "Por que a env `${REDIS_CONNECTION_STRING}` no
`feature_store.yaml`?" — Para o mesmo YAML funcionar em local (`localhost`), Docker
(`redis`) e CI sem editar nada.

---

## Checklist final de domínio

Você domina o projeto se consegue, de caderno fechado:

- [ ] desenhar a arquitetura e nomear o papel de cada caixa;
- [ ] explicar offline vs. online com a tabela de trade-offs;
- [ ] definir point-in-time correctness e apontar as **duas** linhas de código que o
      garantem (`shift(1)` e o as-of join);
- [ ] explicar materialização e quando ela roda em produção;
- [ ] distinguir feature view de on-demand feature view com um exemplo do projeto;
- [ ] justificar cada trade-off (Parquet, Redis, RandomForest, split temporal);
- [ ] dizer por que acurácia é a métrica errada aqui;
- [ ] apontar a maior fraqueza do projeto e como você a resolveria.

Quando todos os itens estiverem marcados, releia o
[`STUDY_GUIDE.md`](STUDY_GUIDE.md) e responda às perguntas em voz alta. Se travar em
alguma, volte à etapa correspondente deste roteiro.
