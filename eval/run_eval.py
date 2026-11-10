"""Retrieval evaluation: vector-only vs BM25-only vs hybrid (RRF) search.

Usage:
    python eval/fetch_corpus.py        #  downloads the CDC guideline corpus
    python eval/run_eval.py           
"""

import argparse
import hashlib
import json
import re
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

HERE = Path(__file__).parent
KS = (1, 3, 5, 10)
CANDIDATES = 50  # depth of each ranked list fed into fusion
EMBED_MODEL = "gemini-embedding-2-preview"  # same model as ingestion_pipeline.py

load_dotenv(HERE.parent / ".env")


# ----------------------------------------------------------------- data ----
def load_corpus():
    docs = {}
    for p in sorted((HERE / "corpus").glob("*.txt")):
        docs[p.stem] = p.read_text(encoding="utf-8")
    if not docs:
        raise SystemExit("eval/corpus is empty - run `python eval/fetch_corpus.py` first")
    return docs


def chunk_corpus(docs, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    chunks = []
    for name, text in docs.items():
        for i, c in enumerate(splitter.split_documents([Document(page_content=text, metadata={"doc": name})])):
            c.metadata["chunk_id"] = f"{name}:{i}"
            c.metadata["end_index"] = c.metadata["start_index"] + len(c.page_content)
            chunks.append(c)
    return chunks


def load_questions(docs):
    """Load questions and resolve each evidence snippet to character spans."""
    questions = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))
    for q in questions:
        text = docs[q["doc"]]
        spans = []
        for ev in q["evidence"]:
            found = [m.start() for m in re.finditer(re.escape(ev), text)]
            if not found:
                raise SystemExit(f"{q['id']}: evidence not found verbatim in {q['doc']}: {ev!r}")
            spans += [(s, s + len(ev)) for s in found]
        q["spans"] = spans
    return questions


def is_relevant(meta, q):
    if meta["doc"] != q["doc"]:
        return False
    for s, e in q["spans"]:
        overlap = min(e, meta["end_index"]) - max(s, meta["start_index"])
        if overlap >= 0.5 * (e - s):
            return True
    return False


# ------------------------------------- retrievers ---------------------------
STOPWORDS = set(
    "a an and are as at be by can do does for from has have how i if in is it its of on or should "
    "that the their they this to was were what when where which who why will with without must".split()
)
USE_STOPWORDS = False  # set by --bm25-stopwords


def tokenize(text):
    toks = re.findall(r"\w+", text.lower())
    return [t for t in toks if t not in STOPWORDS] if USE_STOPWORDS else toks


class Retrievers:
    def __init__(self, chunks, store):
        self.chunks = chunks
        self.store = store
        self.bm25 = BM25Okapi([tokenize(c.page_content) for c in chunks])
        self.by_id = {c.metadata["chunk_id"]: c for c in chunks}
        self._qcache = {}

    def embed_query(self, query):
        """Embed each query once (cached); back off on free-tier rate limits."""
        if query not in self._qcache:
            for attempt in range(8):
                try:
                    self._qcache[query] = self.store._embedding_function.embed_query(query)
                    break
                except Exception as e:
                    wait = min(60, 5 * 2**attempt)
                    print(f"  query embed failed ({type(e).__name__}); retry in {wait}s")
                    time.sleep(wait)
            else:
                raise SystemExit("query embedding failed repeatedly")
        return self._qcache[query]

    def vector(self, query, n=CANDIDATES):
        res = self.store.similarity_search_by_vector(self.embed_query(query), k=n)
        return [d.metadata["chunk_id"] for d in res]

    def bm25_search(self, query, n=CANDIDATES):
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return [self.chunks[i].metadata["chunk_id"] for i in order if scores[i] > 0]

    def hybrid(self, query, n=CANDIDATES, rrf_k=60):
        return rrf([self.vector(query, n), self.bm25_search(query, n)], rrf_k)


def rrf(rankings, k=60):
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank)."""
    score = defaultdict(float)
    for ranking in rankings:
        for rank, cid in enumerate(ranking, 1):
            score[cid] += 1.0 / (k + rank)
    return sorted(score, key=score.get, reverse=True)


# -------------------------- vectors -----------------------------------------
def build_store(chunks, chunk_size, chunk_overlap):
    """Embed the chunks once; the Chroma collection is cached on disk by config hash."""
    tag = hashlib.md5(f"{EMBED_MODEL}|{chunk_size}|{chunk_overlap}|{len(chunks)}".encode()).hexdigest()[:10]
    emb = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
    store = Chroma(
        collection_name=f"eval_{tag}",
        persist_directory=str(HERE / "chroma_eval"),
        embedding_function=emb,
        collection_metadata={"hnsw:space": "cosine"},
    )
    have = store._collection.count()
    if have >= len(chunks):
        print(f"Reusing cached embeddings ({have} chunks, collection eval_{tag})")
        return store
    if have:
        store.delete_collection()
        store = Chroma(
            collection_name=f"eval_{tag}",
            persist_directory=str(HERE / "chroma_eval"),
            embedding_function=emb,
            collection_metadata={"hnsw:space": "cosine"},
        )
    print(f"Embedding {len(chunks)} chunks with {EMBED_MODEL} ...")
    B = 50
    for i in range(0, len(chunks), B):
        batch = chunks[i : i + B]
        for attempt in range(6):
            try:
                store.add_documents(batch, ids=[c.metadata["chunk_id"] for c in batch])
                break
            except Exception as e:  # rate limit / transient
                wait = 5 * 2**attempt
                print(f"  batch {i // B} failed ({type(e).__name__}); retry in {wait}s")
                time.sleep(wait)
        else:
            raise SystemExit("embedding failed repeatedly")
        print(f"  {min(i + B, len(chunks))}/{len(chunks)}")
    return store


# ---------------------- metrics ---------------------------------------------
def evaluate(name, search, questions, by_id):
    rows = []
    for q in questions:
        ranked = search(q["question"])
        first = next((r for r, cid in enumerate(ranked, 1) if is_relevant(by_id[cid].metadata, q)), None)
        rows.append({"id": q["id"], "doc": q["doc"], "type": q["type"], "first_rank": first})
    return {"name": name, "rows": rows}


def summarize(rows):
    n = len(rows)
    out = {"n": n}
    for k in KS:
        out[f"hit@{k}"] = sum(1 for r in rows if r["first_rank"] and r["first_rank"] <= k) / n
    out["mrr@10"] = sum(1 / r["first_rank"] for r in rows if r["first_rank"] and r["first_rank"] <= 10) / n
    return out


def fmt_table(results, questions, group_key=None):
    systems = [r["name"] for r in results]
    groups = {"all": lambda r: True}
    if group_key:
        vals = sorted({q[group_key] for q in questions})
        groups = {v: (lambda r, v=v: r[group_key] == v) for v in vals}
    lines = []
    for gname, pred in groups.items():
        lines.append(f"\n**{gname}**\n")
        lines.append("| system | n | " + " | ".join(f"Hit@{k}" for k in KS) + " | MRR@10 |")
        lines.append("|---|---|" + "---|" * (len(KS) + 1))
        for res in results:
            s = summarize([r for r in res["rows"] if pred(r)])
            lines.append(
                f"| {res['name']} | {s['n']} | "
                + " | ".join(f"{s[f'hit@{k}']:.2f}" for k in KS)
                + f" | {s['mrr@10']:.3f} |"
            )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-size", type=int, default=1000)
    ap.add_argument("--chunk-overlap", type=int, default=150)
    ap.add_argument("--bm25-stopwords", action="store_true", help="drop English stopwords for BM25 (exploratory)")
    args = ap.parse_args()
    global USE_STOPWORDS
    USE_STOPWORDS = args.bm25_stopwords
    suffix = "_bm25_stopwords" if args.bm25_stopwords else ""

    docs = load_corpus()
    questions = load_questions(docs)
    chunks = chunk_corpus(docs, args.chunk_size, args.chunk_overlap)
    print(f"{len(docs)} docs -> {len(chunks)} chunks; {len(questions)} questions")
    # sanity: every question must have at least one relevant chunk
    for q in questions:
        assert any(is_relevant(c.metadata, q) for c in chunks), f"{q['id']} has no relevant chunk"

    store = build_store(chunks, args.chunk_size, args.chunk_overlap)
    R = Retrievers(chunks, store)

    results = [
        evaluate("vector-only", R.vector, questions, R.by_id),
        evaluate("bm25-only", R.bm25_search, questions, R.by_id),
        evaluate("hybrid (RRF)", R.hybrid, questions, R.by_id),
    ]

    # paired comparison at k=5: where do hybrid and vector disagree?
    v = {r["id"]: r["first_rank"] for r in results[0]["rows"]}
    h = {r["id"]: r["first_rank"] for r in results[2]["rows"]}
    hit = lambda x, k=5: x is not None and x <= k
    hybrid_only = [i for i in v if hit(h[i]) and not hit(v[i])]
    vector_only = [i for i in v if hit(v[i]) and not hit(h[i])]
    both_miss = [i for i in v if not hit(v[i]) and not hit(h[i])]

    md = [
        "# Retrieval evaluation results",
        f"\nCorpus: {len(docs)} CDC MMWR guideline documents, {len(chunks)} chunks "
        f"(recursive splitter, size {args.chunk_size}, overlap {args.chunk_overlap}). "
        f"Embeddings: `{EMBED_MODEL}`. {len(questions)} questions. "
        f"Hybrid = RRF(k=60) over top-{CANDIDATES} vector + top-{CANDIDATES} BM25. "
        f"BM25 stopword removal: {'on' if USE_STOPWORDS else 'off'}.",
        "\n## Overall",
        fmt_table(results, questions),
        "\n## By question type",
        fmt_table(results, questions, "type"),
        "\n## By document",
        fmt_table(results, questions, "doc"),
        "\n## Paired comparison at k=5 (hybrid vs vector-only)",
        f"\n- hybrid hit, vector missed: {len(hybrid_only)} {hybrid_only}",
        f"- vector hit, hybrid missed: {len(vector_only)} {vector_only}",
        f"- both missed: {len(both_miss)} {both_miss}",
    ]
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"results{suffix}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out_dir / f"results{suffix}.json").write_text(
        json.dumps({"config": vars(args), "results": results}, indent=2), encoding="utf-8"
    )
    print("\n".join(md))


if __name__ == "__main__":
    main()
