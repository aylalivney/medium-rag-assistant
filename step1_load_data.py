import pandas as pd
import json

CSV_PATH = "data/medium-english-50mb.csv"
OUTPUT_PATH = "processed/chunks_sample.json"

CHUNK_SIZE = 512
OVERLAP = 100


def split_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    words = str(text).split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        if chunk_text.strip():
            chunks.append(chunk_text)

        start += chunk_size - overlap

    return chunks


df = pd.read_csv(CSV_PATH)
df = df.head(200).copy()
df["article_id"] = df.index

all_chunks = []

for _, row in df.iterrows():
    article_chunks = split_into_chunks(row["text"])

    for chunk_index, chunk_text in enumerate(article_chunks):
        all_chunks.append({
            "article_id": int(row["article_id"]),
            "chunk_index": chunk_index,
            "title": row["title"],
            "authors": row["authors"],
            "tags": row["tags"],
            "url": row["url"],
            "timestamp": row["timestamp"],
            "text": chunk_text
        })

print("Number of articles:", len(df))
print("Number of chunks:", len(all_chunks))

print("\nExample chunk:")
print(json.dumps(all_chunks[0], indent=2, ensure_ascii=False)[:1500])

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)

print(f"\nSaved chunks to {OUTPUT_PATH}")