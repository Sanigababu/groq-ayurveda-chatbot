import os
import json
import traceback
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb

# Load environment variables
print("🔄 Starting application setup...")
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


# Validate API Key
if not GROQ_API_KEY:
    print("❌ GROQ_API_KEY is missing!")
HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

# Prompt instructions
prompt_instructions = """
You are an Ayurvedic Assistant, an expert in traditional Indian medicine and holistic wellness.
Your goal is to provide helpful, balanced, and culturally respectful Ayurvedic suggestions based on user symptoms.
Always ensure your advice is safe, herbal, and suggests seeing a practitioner when needed.
"""

# Initialize FastAPI
app = FastAPI()

try:
    print("✅ Dependencies loaded")
    print("🔁 Loading model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("🧠 Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path="./chroma_store")
    collection = client.get_or_create_collection("chat_chunks")

    print("🚀 FastAPI App is ready!")

except Exception as e:
    print("❌ Startup error:", str(e))
    traceback.print_exc()

# Request schema
class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(chat_req: ChatRequest):
    try:
        message = chat_req.message
        print(f"🗨️ Received message: {message}")

        # Vectorize and search ChromaDB
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

        # Send to Groq
        res = requests.post(GROQ_API_URL, headers=HEADERS, json={
            "model": "llama3-8b-8192",
            "messages": messages
        })

        print("🔁 Groq response:", res.status_code, res.text)
        res.raise_for_status()
        reply = res.json()["choices"][0]["message"]["content"]
        return {"reply": reply}

    except Exception as e:
        print("❌ Error:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}

