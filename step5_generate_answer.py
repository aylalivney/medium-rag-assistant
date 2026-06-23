import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone

load_dotenv()

QUESTION = "List exactly 3 articles about writing. Return only the titles."

SYSTEM_PROMPT = """
You are a Medium-article assistant that answers questions strictly and only
based on the Medium articles dataset context provided to you.
You must not use any external knowledge, the open internet, or information
that is not explicitly contained in the retrieved context.

If the answer cannot be determined from the provided context, respond:
“I don’t know based on the provided Medium articles data.”

Always explain your answer using the given context, quoting or paraphrasing
the relevant article passage or metadata when helpful.
"""

embeddings = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL"),
    api_key=os.getenv("LLMOD_API_KEY"),
    base_url=os.getenv("LLMOD_BASE_URL"),
)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

question_vector = embeddings.embed_query(QUESTION)

results = index.query(
    vector=question_vector,
    top_k=5,
    include_metadata=True
)

context_parts = []

for i, match in enumerate(results["matches"], start=1):
    md = match["metadata"]
    context_parts.append(
        f"""
Context {i}
Title: {md.get("title")}
Authors: {md.get("authors")}
URL: {md.get("url")}
Chunk index: {md.get("chunk_index")}
Score: {match.get("score")}
Text:
{md.get("text")}
"""
    )

context = "\n\n".join(context_parts)

user_prompt = f"""
Question:
{QUESTION}

Retrieved context:
{context}

Answer the question using only the retrieved context.
"""

chat = ChatOpenAI(
    model="NBUECSE-gpt-5-mini",
    api_key=os.getenv("LLMOD_API_KEY"),
    base_url=os.getenv("LLMOD_BASE_URL"),
)

response = chat.invoke([
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": user_prompt}
])

print("Question:")
print(QUESTION)

print("\nAnswer:")
print(response.content)
