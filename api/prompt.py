from http.server import BaseHTTPRequestHandler
import json
import os

from dotenv import load_dotenv

load_dotenv()

TOP_K = 7

SYSTEM_PROMPT = """
You are a Medium-article assistant that answers questions strictly and only
based on the Medium articles dataset context provided to you. You must not use
external knowledge or information not contained in the retrieved context.
If the answer cannot be determined from the provided context, respond:
I don't know based on the provided Medium articles data.
Always explain your answer using the given context.
"""


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        response = {
            "chunk_size": 512,
            "overlap_ratio": 0.2,
            "top_k": 7
        }

        self.send_response(200)
        self.send_header("Content-type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode("utf-8"))

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            from langchain_openai import OpenAIEmbeddings, ChatOpenAI
            from pinecone import Pinecone
            data = json.loads(body)
            question = data.get("question", "")
            if not question:
                raise ValueError("Missing question field")

            embeddings = OpenAIEmbeddings(
                model=os.getenv("EMBEDDING_MODEL"),
                api_key=os.getenv("LLMOD_API_KEY"),
                base_url=os.getenv("LLMOD_BASE_URL"),
            )

            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

            question_vector = embeddings.embed_query(question)

            results = index.query(
                vector=question_vector,
                top_k=TOP_K,
                include_metadata=True
            )

            context = []

            for match in results["matches"]:
                md = match["metadata"]
                context.append({
                    "article_id": str(md.get("article_id")),
                    "title": md.get("title"),
                    "chunk": md.get("text"),
                    "score": match.get("score")
                })

            context_text = "\n\n".join([
                f"""
Title: {item["title"]}
Article ID: {item["article_id"]}
Score: {item["score"]}
Chunk:
{item["chunk"]}
"""
                for item in context
            ])

            user_prompt = f"""
Question:
{question}

Retrieved context:
{context_text}

Answer the question using only the retrieved context.
"""

            chat = ChatOpenAI(
                model=os.getenv("CHAT_MODEL"),
                api_key=os.getenv("LLMOD_API_KEY"),
                base_url=os.getenv("LLMOD_BASE_URL"),
            )

            response = chat.invoke([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ])

            output = {
                "response": response.content,
                "context": context,
                "Augmented_prompt": {
                    "System": SYSTEM_PROMPT,
                    "User": user_prompt
                }
            }

            self.send_response(200)
            self.send_header("Content-type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(output, ensure_ascii=False).encode("utf-8"))

        except Exception as e:
            self.send_response(500)
            self.send_header("Content-type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))