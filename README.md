# rag-router — Retrieve / Don't-retrieve / Abstain

A reproducible research codebase for the question: **for each query, should a RAG system
(1) answer from the model's own knowledge, (2) retrieve first, or (3) abstain?**

```
question ──► router ──► no_retrieve → LLM answers closed-book
                    ├─► retrieve    → BM25 → LLM answers with passages
                    └─► abstain     → "I don't know"
```

## What is evaluated (all on a held-out test split)

| Method | Meaning |
|---|---|
| `never_retrieve` | pure parametric baseline |
| `always_retrieve` | standard RAG baseline |
| `rag+conf_threshold` | **strong baseline**: always RAG, abstain if answer confidence < τ (τ tuned on dev) |
| `router[...]` (5 variants) | trained classifiers; the bracket lists which feature groups they see |
| `router[distilbert:query]` | optional tiny transformer that sees only the query text |
| `ORACLE` | always picks the best action in hindsight (upper bound) |

Feature groups: `query` (length, wh-word, capitals…), `param` (closed-book answer confidence),
`retr` (BM25 score stats, query-term coverage), `rag` (confidence of a full RAG answer — post-hoc, expensive;
included to show how much information you *could* have).

**Utility** (configurable): correct answer +1, wrong answer −0.25, retrieval costs 0.05,
abstaining is +1 on unanswerable questions and 0 on answerable ones.
We also report accuracy, hallucination rate, abstain rate, false-abstain rate, and
**LLM calls / retrieval calls per query** (including the cost of computing the router's own inputs).
Every number has a bootstrap 95% CI, and routers get a *paired* CI of the difference vs. `always_retrieve`.

## Step-by-step

### Step 0 — verify the install offline (30 s, no GPU, no internet)
```bash
pip install -e ".[dev,plot]"
pytest -q                                              # 13 tests
python -m rag_router all --config configs/synthetic_demo.yaml
```
This runs the whole pipeline on a toy world with a mock LLM. **Its numbers mean nothing scientifically**;
it only proves the plumbing works.

### Step 1 — set up Colab (Runtime → T4 GPU)
```python
!git clone <your-repo-url> rag-router  # or upload the folder / unzip it
%cd rag-router
!pip install -q -e ".[hf,plot]"
```

### Step 2 — smoke-test with a real model (~3 min)
```python
!python -m rag_router all --config configs/squad_smoke.yaml
```

### Step 3 — the real experiment
```python
!python -m rag_router all --config configs/squad_flan_t5_base.yaml
```
Or run the stages one at a time (each reads the previous stage's files, so you can re-run cheaply):

| Stage | Command | What it does | Output (`runs/<name>/…`) |
|---|---|---|---|
| prepare | `python -m rag_router prepare --config C` | loads SQuAD 2.0 dev, samples 50/50 answerable/unanswerable, builds the passage corpus, splits **by source passage** into 60/20/20 | `data/questions.jsonl`, `data/corpus.jsonl` |
| label | `… label --config C` | BM25 top-k, then Flan-T5 **without** and **with** retrieval; records correctness, confidence, features | `data/outcomes.csv` |
| train | `… train --config C` | derives the oracle action per training question, fits the routers | `routers/*.joblib` |
| evaluate | `… evaluate --config C` | scores every method on the test split | `results/results.md`, `results.csv`, `diagnostics.json`, `utility.png` |

`label` is the only slow stage. Changing `retrieval_cost`, `wrong_penalty`, or the router model only needs `train` + `evaluate`.
Override anything inline: `--set n_questions=500 top_k=5 router_model=logreg`.

### Step 4 — ablations worth running (each is a paper section)
1. `top_k` ∈ {1, 3, 5}; 2. `generator` ∈ {flan-t5-small, base, large} (does a bigger model need retrieval less?);
3. `retriever=dense`; 4. `wrong_penalty` ∈ {0, 0.25, 1} (how risk-averse the system must be);
5. `use_distilbert=true` (the low-compute pivot); 6. a second domain (below).

### Adapting to a new domain (legal / medical)
Create a JSONL with `{"question", "answers": [...], "context": "..."}` per line; empty `answers` = unanswerable.
Copy `configs/custom_domain_template.yaml`, set `questions_path`, run `all`. Unanswerable
items can be made by pairing a question with a passage that does not contain the answer, as SQuAD 2.0 did.

## Reliability features
- Every stage seeded; a test asserts two runs give identical results.
- Splits grouped by source passage (no near-duplicate leakage); thresholds/classifiers fit on train/dev only, test used once.
- Official SQuAD answer normalisation; configurable F1 correctness threshold.
- Config typos raise errors; `config_used.json` is saved with every run.
- Oracle is checked in tests to dominate every fixed policy.

## Read this before you write the paper (known limitations)
1. **SQuAD "unanswerable" ≠ "not in corpus".** Those questions were written adversarially about a *specific paragraph*.
   Here the corpus is all dev passages, so such a question usually retrieves its source paragraph and a plausible-but-wrong
   span — a hard, realistic case, but a rare unanswerable question may be answerable from another passage. Say so.
2. **Expect few `no_retrieve` labels on SQuAD.** Questions are tied to specific passages, so closed-book accuracy is low and the
   parametric class will be small. The router mostly separates *retrieve* from *abstain*. Report the label distribution
   (it's in `diagnostics.json`). To study the parametric class properly, add a second domain whose questions a model can know (e.g. trivia
   with a Wikipedia index).
3. **Natural Questions:** `nq_open` has no passages, so it needs a Wikipedia index to retrieve from. That is a bigger build than a drop-in
   dataset, so it is not included; use the `jsonl` loader if you build one.
4. **`router[...+rag]` is not a deployable system** — it needs a full RAG call per query. It is there as an information upper-ish bound.
5. The oracle depends on your utility weights; report them, and show the sensitivity ablation.
6. Confidence = exp(mean token log-prob) from a small seq2seq model — known to be imperfectly calibrated; that is part of the story.
7. Untested here: the Hugging Face generator, dense retriever and DistilBERT router (the build sandbox had no access to
   huggingface.co). They compile and follow standard APIs, but run the Step-2 smoke test first.
