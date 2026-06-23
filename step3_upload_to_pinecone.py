import json
import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

load_dotenv()

CHUNKS_PATH = "processed/chunks_sample.json"
BATCH_SIZE = 50

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks")

embeddings = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL"),
    api_key=os.getenv("LLMOD_API_KEY"),
    base_url=os.getenv("LLMOD_BASE_URL"),
)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

for start in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[start:start + BATCH_SIZE]

    texts = [chunk["text"] for chunk in batch]

    # Embedding in batch - important for the assignment
    vectors = embeddings.embed_documents(texts)

    vectors_to_upload = []

    for chunk, vector in zip(batch, vectors):
        vectors_to_upload.append({
            "id": f'{chunk["article_id"]}_{chunk["chunk_index"]}',
            "values": vector,
            "metadata": {
                "article_id": chunk["article_id"],
                "title": chunk["title"],
                "authors": chunk["authors"],
                "tags": chunk["tags"],
                "url": chunk["url"],
                "timestamp": chunk["timestamp"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"]
            }
        })

    index.upsert(vectors=vectors_to_upload)

    print(f"Uploaded batch {start} to {start + len(batch)}")

print("All chunks uploaded successfully!")