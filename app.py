import os
import json
import gradio as gr
import requests
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

# System prompt for Ayurvedic Assistant
prompt_instructions = """
You are an Ayurvedic Assistant, an expert in traditional Indian medicine and holistic wellness.

Guidelines:
- Focus only on Ayurvedic health-related advice.
- Start with a warm, brief greeting.
- Ask follow-up or clarifying questions before giving advice.
- Keep responses short (max one paragraph or 3–4 sentences).
- Be conversational, natural, and engaging like ChatGPT.
- Base your advice on Ayurvedic principles—use natural remedies, herbs, dietary/lifestyle changes.
- If appropriate, briefly explain relevant Ayurvedic concepts like doshas (Vata, Pitta, Kapha), dhatus, etc.
- Use Sanskrit terms only when helpful and explain them.
- Be warm, compassionate, and respectful.
- Never give advice without knowing enough symptoms.
- Emphasize this is not a replacement for professional medical care.
"""

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load ChromaDB
client = chromadb.PersistentClient(path="./chroma_store")
collection = client.get_or_create_collection("chat_chunks")

# Search function
def search_docs(query: str, top_k: int = 3, max_doc_chars: int = 300):
    query_embedding = model.encode([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )
    documents = results.get("documents", [[]])[0]
    return [doc[:max_doc_chars] for doc in documents if doc]

# Chat function
def chat_fn(message, history):
    # Add user message to history
    history = history or []
    history.append(("user", message))

    # Get context from ChromaDB
    docs = search_docs(message)
    context = "\n".join(docs)[:3000]

    # Build message list for Groq API
    messages = [{"role": "system", "content": prompt_instructions.strip()}]
    for role, msg in history:
        messages.append({"role": role, "content": msg})
    messages.insert(1, {
        "role": "system",
        "content": f"Use the following Ayurvedic knowledge base for reference only. Do not quote or reference directly:\n\n{context}"
    })

    # Send to Groq
    try:
        response = requests.post(
            GROQ_API_URL,
            headers=HEADERS,
            json={"model": "llama3-8b-8192", "messages": messages}
        )
        reply = response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        reply = f"❌ Error: {e}"

    # Add assistant reply to history
    history.append(("assistant", reply))
    return reply, history

# Interface
gr.ChatInterface(chat_fn, title="🌿 Ayurvedic Assistant", theme="soft").launch()
