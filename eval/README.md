# Retrieval evaluation: vector vs BM25 vs hybrid

Question: on real clinical-guideline text, how often does each retriever return the
passage that actually contains the answer?

## Setup

- **Corpus** - 4 public-domain CDC MMWR documents (US government works), fetched by
  `fetch_corpus.py`: HBV screening 2023, HCV screening 2020, Dengvaxia ACIP
  recommendations 2021, PHS guideline on HIV/HBV/HCV in solid organ donors 2020.
  ~263k characters -> 454 chunks (recursive splitter, 1000 chars, 150 overlap).
  The corpus is git-ignored; rerun the fetch script to rebuild it.
- **Questions** - 46 hand-written questions in `questions.json` (11 HBV, 11 HCV,
  12 dengue, 12 organ donor). Each has a verbatim `evidence` snippet from the
  source document. A retrieved chunk is correct if it overlaps that evidence by >=50%,
  so labels do not depend on chunking choices. The loader fails if any snippet is
  not found verbatim.
  Each question is tagged `keyword` (shares rare terms, numbers or acronyms with
  the passage) or `semantic` (paraphrased). These tags are my judgment.
- **Systems** - vector-only (Chroma, cosine, `gemini-embedding-2-preview`, same
  model as `ingestion_pipeline.py`); BM25-only (`rank-bm25`, Okapi); hybrid =
  Reciprocal Rank Fusion (k=60) of the top-50 from each.
- **Metrics** - Hit@k (a relevant chunk is in the top k; this equals recall@k
  here because each question has one answer location) and MRR@10.

Run: `python eval/fetch_corpus.py && python eval/run_eval.py`
(add `--bm25-stopwords` for the second experiment).

## Results (n = 46)

Main run, plain BM25 tokenization (`results/results.md`):

| system | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 |
|---|---|---|---|---|---|
| vector-only | 0.74 | 0.89 | 0.96 | 0.96 | 0.812 |
| bm25-only | 0.50 | 0.67 | 0.78 | 0.80 | 0.608 |
| hybrid (RRF) | 0.70 | 0.87 | 0.91 | 0.93 | 0.795 |

Exploratory run, BM25 with stopwords removed (`results/results_bm25_stopwords.md`):

| system | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 |
|---|---|---|---|---|---|
| vector-only | 0.74 | 0.89 | 0.96 | 0.96 | 0.812 |
| bm25-only | 0.57 | 0.72 | 0.74 | 0.80 | 0.644 |
| hybrid (RRF) | 0.67 | 0.80 | 0.89 | 0.96 | 0.761 |

By question type (main run, Hit@5): keyword - vector 0.95, BM25 0.89, hybrid 0.95;
semantic - vector 0.96, BM25 0.70, hybrid 0.89.

## What this shows

- **Hybrid did not beat vector-only on this set.** At k=5 it recovered 1 question
  vector missed and lost 3 (4 in the stopword run). With 46 questions that gap is
  within noise, so the fair reading is "no measurable gain", not "hybrid is worse".
- BM25 is competitive on keyword-style questions (Hit@1 0.74 vs 0.68 for vector in
  the main run) and weak on paraphrased ones (Hit@5 0.70 vs 0.96). Fusing in a
  weak ranker dilutes a strong one under equal-weight RRF.
- Stopword removal lifts BM25-only (MRR 0.61 -> 0.64) but does not change the hybrid conclusion.
- Both vector and hybrid miss `dg05` (immunocompromised children / live vaccine
  contraindication) - a paraphrase the embedding model ranks 30th.

## Caveats

- Small, single-author question set; no confidence intervals. Treat differences of
  a few questions as noise.
- The stopword run was added after seeing the main results, on the same questions,
  so it is exploratory. Do not tune further against these 46 questions.
- The questions are written from the evidence, which favours lexical overlap for
  `keyword` items and was not independently validated.
- Four documents from one publisher; the vector model is a single strong embedder.
  Results may differ with weaker embeddings, larger corpora, or messier internal docs.

## Next experiments

Weighted fusion / tuned RRF `k` on a held-out split, a cross-encoder reranker on the
hybrid candidates, chunk size and the project's three chunking strategies, and a
larger question set with a second labeller.
