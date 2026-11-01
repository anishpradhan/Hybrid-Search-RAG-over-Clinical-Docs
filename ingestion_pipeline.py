import os
from annotated_types import doc
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
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

    loader = DirectoryLoader(path=docs_path, glob="*.txt", loader_cls=TextLoader)

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


def split_documents(documents, chunk_size=800, chunk_overlap=0):
    """Split documents into smaller chunks with overlap."""
    print(
        f"\nSplitting documents into chunks of size {chunk_size} with overlap {chunk_overlap}..."
    )

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
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


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store from"""
    print("Creating embeddings and storing in ChromaDB...")

    # embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

    # Create ChromaDB vector store
    print("--- Creating vector store --- ")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )

    print("--- Finished creating vector store ---")

    print("Vector store created and saved to {persist_directory}")

    return vectorstore


def main():
    # 1 Load documents
    documents = load_documents(docs_path="docs")

    # 2 Split documents into chunks
    chunks = split_documents(documents, chunk_size=800, chunk_overlap=0)

    create_vector_store(chunks)
if __name__ == "__main__":
    
    main()
