import json
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

CHUNKS_PATH = "processed/chunks_sample.json"

embeddings = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL"),
    api_key=os.getenv("LLMOD_API_KEY"),
    base_url=os.getenv("LLMOD_BASE_URL"),
)

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

texts = [chunk["text"] for chunk in chunks[:3]]

vectors = embeddings.embed_documents(texts)

print("Number of vectors:", len(vectors))
print("Vector dimension:", len(vectors[0]))
print("First 5 numbers of first vector:")
print(vectors[0][:5])