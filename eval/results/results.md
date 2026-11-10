# Retrieval evaluation results

Corpus: 4 CDC MMWR guideline documents, 454 chunks (recursive splitter, size 1000, overlap 150). Embeddings: `gemini-embedding-2-preview`. 46 questions. Hybrid = RRF(k=60) over top-50 vector + top-50 BM25.

## Overall

**all**

| system | n | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 |
|---|---|---|---|---|---|---|
| vector-only | 46 | 0.74 | 0.89 | 0.96 | 0.96 | 0.812 |
| bm25-only | 46 | 0.50 | 0.67 | 0.78 | 0.80 | 0.608 |
| hybrid (RRF) | 46 | 0.70 | 0.87 | 0.91 | 0.93 | 0.795 |

## By question type

**keyword**

| system | n | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 |
|---|---|---|---|---|---|---|
| vector-only | 19 | 0.68 | 0.84 | 0.95 | 0.95 | 0.761 |
| bm25-only | 19 | 0.74 | 0.89 | 0.89 | 0.89 | 0.816 |
| hybrid (RRF) | 19 | 0.68 | 0.84 | 0.95 | 1.00 | 0.794 |

**semantic**

| system | n | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 |
|---|---|---|---|---|---|---|
| vector-only | 27 | 0.78 | 0.93 | 0.96 | 0.96 | 0.849 |
| bm25-only | 27 | 0.33 | 0.52 | 0.70 | 0.74 | 0.461 |
| hybrid (RRF) | 27 | 0.70 | 0.89 | 0.89 | 0.89 | 0.796 |

## By document

**dengue_vaccine_2021**

| system | n | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 |
|---|---|---|---|---|---|---|
| vector-only | 12 | 0.58 | 0.67 | 0.83 | 0.83 | 0.649 |
| bm25-only | 12 | 0.58 | 0.67 | 0.75 | 0.83 | 0.658 |
| hybrid (RRF) | 12 | 0.67 | 0.75 | 0.92 | 0.92 | 0.746 |

**hepatitis_b_screening_2023**

| system | n | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 |
|---|---|---|---|---|---|---|
| vector-only | 11 | 0.82 | 1.00 | 1.00 | 1.00 | 0.894 |
| bm25-only | 11 | 0.36 | 0.55 | 0.73 | 0.73 | 0.480 |
| hybrid (RRF) | 11 | 0.64 | 0.91 | 0.91 | 0.91 | 0.773 |

**hepatitis_c_screening_2020**

| system | n | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 |
|---|---|---|---|---|---|---|
| vector-only | 11 | 0.73 | 0.91 | 1.00 | 1.00 | 0.811 |
| bm25-only | 11 | 0.64 | 0.82 | 0.82 | 0.82 | 0.727 |
| hybrid (RRF) | 11 | 0.82 | 1.00 | 1.00 | 1.00 | 0.909 |

**organ_donor_screening_2020**

| system | n | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 |
|---|---|---|---|---|---|---|
| vector-only | 12 | 0.83 | 1.00 | 1.00 | 1.00 | 0.903 |
| bm25-only | 12 | 0.42 | 0.67 | 0.83 | 0.83 | 0.565 |
| hybrid (RRF) | 12 | 0.67 | 0.83 | 0.83 | 0.92 | 0.762 |

## Paired comparison at k=5 (hybrid vs vector-only)

- hybrid hit, vector missed: 1 ['dg09']
- vector hit, hybrid missed: 3 ['hb02', 'og06', 'og07']
- both missed: 1 ['dg05']
