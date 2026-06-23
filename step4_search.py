import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

load_dotenv()

embeddings = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL"),
    api_key=os.getenv("LLMOD_API_KEY"),
    base_url=os.getenv("LLMOD_BASE_URL"),
)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

question = "How does coronavirus affect the brain?"

question_vector = embeddings.embed_query(question)

results = index.query(
    vector=question_vector,
    top_k=5,
    include_metadata=True
)

print("Question:", question)
print("\nTop results:")

for match in results["matches"]:
    print("\n---")
    print("Score:", match["score"])
    print("Title:", match["metadata"].get("title"))
    print("Authors:", match["metadata"].get("authors"))
    print("URL:", match["metadata"].get("url"))
    print("Chunk index:", match["metadata"].get("chunk_index"))
    print("Text preview:", match["metadata"].get("text", "")[:500])
