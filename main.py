import os
import json
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

# Prompt instructions
prompt_instructions = """
You are an Ayurvedic Assistant, an expert in traditional Indian medicine and holistic wellness.
... (same instructions from before)
"""

# Load model and ChromaDB
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_store")
collection = client.get_or_create_collection("chat_chunks")

# FastAPI setup
app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(chat_req: ChatRequest):
    message = chat_req.message

    # Search ChromaDB
    query_embedding = model.encode([message])[0]
    results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=3)
    docs = results.get("documents", [[]])[0]
    context = "\n".join([doc[:300] for doc in docs if doc])[:3000]

    # Prepare message payload
    messages = [
        {"role": "system", "content": prompt_instructions.strip()},
        {"role": "system", "content": f"Use the following Ayurvedic knowledge base for reference only:\n\n{context}"},
        {"role": "user", "content": message}
    ]

    try:
        res = requests.post(GROQ_API_URL, headers=HEADERS, json={
            "model": "llama3-8b-8192",
            "messages": messages
        })
        reply = res.json()["choices"][0]["message"]["content"]
        return {"reply": reply}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
