from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
persistent_directory = "db/chroma_db"


# Load embeddings and vector store
embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"},
)

# Search for relevant documents
# query = "Tell me about SpaceX company"
# query = "What is CUDA?"
query = "How does microsoft earn revenue? "

retriever = db.as_retriever(search_kwargs={"k": 1})

# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={"k": 5, "score_threshold": 0.3},  
# )  # Only return chunks with cosine similarity >= 0.3


relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")

# Display results
print("--- Context ---")

for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")


# def main():
#     pass


# if __name__ == "__main__":
#     main()
