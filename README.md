# Hybrid Search RAG over Clinical Guidelines

A small retrieval-augmented generation (RAG) project that answers questions from public clinical guidelines, built with LangChain, ChromaDB and Gemini. It started as a tutorial-style RAG pipeline over a few company documents. I swapped the corpus for CDC guideline documents and added an evaluation to find out whether hybrid search (BM25 plus vectors) actually beats plain vector search on this kind of text.

Short version: on my test set, it didn't. The details are below, because that result is the most interesting part of the project.

## What's in here

| File | What it does |
|---|---|
| `ingestion_pipeline.py` | Loads the `.txt` files in `docs/`, splits them into chunks, embeds them and stores them in Chroma. Throttled so it survives the Gemini free tier. |
| `retrieval_pipeline.py` | Runs one query against the store and has Gemini answer from the top 5 chunks. |
| `history_aware_generation.py` | A command-line chat. It rewrites each follow-up question into a standalone one using the chat history, retrieves 3 chunks, then answers. |
| `character_text_splitter.py`, `semantic_chunking.py`, `agentic_chunking.py` | Three small demos of different chunking strategies on the same short passage (character/recursive, embedding-based, LLM-driven). |
| `eval/` | The retrieval evaluation. Has its own README. |

The main pipeline uses **vector search only**. Hybrid search lives in `eval/`, where I use it as a comparison, not as the default.

## The corpus

`docs/` holds four US government documents (CDC MMWR reports, public domain):

- Screening and Testing for Hepatitis B Virus Infection, 2023
- CDC Recommendations for Hepatitis C Screening Among Adults, 2020
- Dengue Vaccine: ACIP Recommendations, 2021
- Assessing Solid Organ Donors and Monitoring Transplant Recipients for HIV, HBV and HCV, 2020

About 260k characters in total. `eval/fetch_corpus.py` downloads and cleans them from the CDC site, so you can rebuild the folder if you delete it.

## Setup

You need Python 3.13 (that's what I used) and a Gemini API key from Google AI Studio.

```bash
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env           # then put your key in GOOGLE_API_KEY
```

Build the vector store, then try it:

```bash
python ingestion_pipeline.py          # ~6 minutes on the free tier
python retrieval_pipeline.py          # one-off question, edit `query` in the file
python history_aware_generation.py    # interactive chat
```

The ingestion is slow on purpose. The free tier allows 100 embedding requests per minute and every chunk is one request, so the script sends batches of 40 about 30 seconds apart, backs off if it still gets a 429, and skips chunks that are already stored. If it stops halfway, run it again and it carries on. If you already have an old `db/chroma_db` from a different set of documents, delete it first or the results will mix.

Chunking is 1000 characters with 150 overlap, using LangChain's recursive splitter. I started with `CharacterTextSplitter`, which only splits on blank lines. These documents barely use blank lines, so it produced chunks of up to 10,000 characters.

## The evaluation

The question: for a question whose answer sits in one specific passage, how often does each retriever put that passage in the top few results?

- **46 questions** I wrote by hand from the documents, each tied to a verbatim evidence sentence. A retrieved chunk counts as correct if it overlaps that sentence, so the labels don't change when I change chunking.
- **Three systems:** vector-only (Gemini embeddings), BM25-only, and hybrid, which merges the two ranked lists with reciprocal rank fusion.
- **Metrics:** Hit@k (is a correct chunk in the top k) and MRR@10.

| system | Hit@1 | Hit@5 | Hit@10 | MRR@10 |
|---|---|---|---|---|
| vector-only | 0.74 | 0.96 | 0.96 | 0.81 |
| BM25-only | 0.50 | 0.78 | 0.80 | 0.61 |
| hybrid (RRF) | 0.70 | 0.91 | 0.93 | 0.80 |

What I take from it:

- Hybrid did not beat vector search here. At k=5 it rescued one question vector missed and lost three that vector got. On 46 questions that is within noise, so I'd say "no measurable gain", not "worse".
- BM25 does well when a question shares rare terms, numbers or acronyms with the passage, and badly when it's a paraphrase. Averaging a weak ranker into a strong one with equal weights drags the strong one down a bit.
- Gemini's embeddings were already good enough on this corpus that there wasn't much left for keyword search to fix.

Caveats: it's a small question set written by one person, on four documents from one publisher. I also tried removing stopwords from BM25 after seeing the first results; that helped BM25 on its own but not the hybrid, and since it was done after looking at the numbers I count it as exploratory. Full tables, per-document breakdowns and the failure cases are in `eval/README.md` and `eval/results/`.

To rerun it:

```bash
python eval/fetch_corpus.py
python eval/run_eval.py
```

