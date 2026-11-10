import hashlib
import os
import re
import time
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


def load_documents(docs_path="docs"):
    """Load all text files from the docs directory."""
    print(f"Loading documents from {docs_path}...")

    # Check if the docs_path exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(
            f"The directory '{docs_path}' does not exist. Please create it and add some text files."
        )

    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(
            f"No text files found in the directory '{docs_path}'. Please add some text files."
        )

    # for i, doc in enumerate(documents):
    #     print(f"\nDocument {i+1}:")
    #     print(f"Source: {doc.metadata['source']}")
    #     print(f"Content Length: {len(doc.page_content)} characters")
    #     print(f"Content preview: {doc.page_content[:100]}...")  # Print first 100 characters
    #     print(f"Metadata: {doc.metadata}")

    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=150):
    """Split documents into smaller chunks with overlap."""
    print(
        f"\nSplitting documents into chunks of size {chunk_size} with overlap {chunk_overlap}..."
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )

    chunks = text_splitter.split_documents(documents)

    # if chunks:
    #     for i, chunk in enumerate(chunks[:5]):
    #         print(f"\n--- Chunk {i + 1}---")
    #         print(f"Content Source: {chunk.metadata['source']}")
    #         print(f"Content Length: {len(chunk.page_content)} characters")
    #         print(f"Content preview: {chunk.page_content}")
    #         print("-" * 50)

    #     if len(chunks) > 5:
    #         print(f"\n...and {len(chunks) - 5} more chunks.")

    return chunks


def _chunk_id(chunk):
    """Deterministic id so re-running ingestion skips chunks that are already stored."""
    key = f"{chunk.metadata.get('source', '')}|{chunk.metadata.get('start_index', '')}|{chunk.page_content}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def create_vector_store(
    chunks,
    persist_directory="db/chroma_db",
    batch_size=40,
    min_seconds_per_batch=30,  # 40 chunks / 30 s = 80 requests/min, under the 100/min free tier
):
    """Embed chunks in throttled batches and persist them in ChromaDB.

    The Gemini free tier allows 100 embedding requests per minute (one per chunk),
    so sending every chunk at once triggers 429 RESOURCE_EXHAUSTED. Batches are
    paced, 429s are retried after the delay the API asks for, and chunks that are
    already stored are skipped, so an interrupted run can simply be re-run.
    """
    print("Creating embeddings and storing in ChromaDB...")

    # embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
    )

    ids = [_chunk_id(c) for c in chunks]
    existing = set(vectorstore.get(ids=ids)["ids"])
    todo = [(i, c) for i, c in zip(ids, chunks) if i not in existing]
    print(f"{len(existing)} chunks already stored, {len(todo)} to embed")

    for start in range(0, len(todo), batch_size):
        batch = todo[start : start + batch_size]
        t0 = time.time()
        for attempt in range(8):
            try:
                vectorstore.add_documents(
                    [c for _, c in batch], ids=[i for i, _ in batch]
                )
                break
            except Exception as e:
                msg = str(e)
                if "RESOURCE_EXHAUSTED" not in msg and "429" not in msg:
                    raise
                m = re.search(r"retry in ([\d.]+)s", msg)
                wait = (float(m.group(1)) if m else 20) + 5 * (attempt + 1)
                print(f"  rate limited; waiting {wait:.0f}s (attempt {attempt + 1}/8)")
                time.sleep(wait)
        else:
            raise RuntimeError("Embedding kept hitting the rate limit; re-run later to resume.")
        done = min(start + batch_size, len(todo))
        print(f"  embedded {done}/{len(todo)}")
        if done < len(todo):
            time.sleep(max(0, min_seconds_per_batch - (time.time() - t0)))

    print(f"Vector store saved to {persist_directory}")
    return vectorstore


def main():
    # 1 Load documents
    documents = load_documents(docs_path="docs")

    # 2 Split documents into chunks
    chunks = split_documents(documents, chunk_size=1000, chunk_overlap=150)

    create_vector_store(chunks)


if __name__ == "__main__":
    main()
