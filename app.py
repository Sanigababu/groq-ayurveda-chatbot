import os
import json
import requests
import streamlit as st
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

# Ayurvedic assistant system prompt
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

# Load SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load ChromaDB
client = chromadb.PersistentClient(path="./chroma_store")
collection = client.get_or_create_collection("chat_chunks")

# Search top docs
def search_docs(query: str, top_k: int = 3, max_doc_chars: int = 300):
    query_embedding = model.encode([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )
    documents = results.get("documents", [[]])[0]
    return [doc[:max_doc_chars] for doc in documents if doc]

# Streamlit App
st.set_page_config(page_title="Ayurvedic Assistant 🌿", layout="wide")
st.title("🌿 Ayurvedic Assistant")

# Chat history in session state
if "history" not in st.session_state:
    st.session_state.history = []

# User input
user_input = st.chat_input("Ask me anything related to Ayurveda...")
if user_input:
    # Store user message
    st.session_state.history.append({"role": "user", "content": user_input})

    # Retrieve context
    docs = search_docs(user_input)
    context = "\n".join(docs)[:3000]

    # Prepare message list
    messages = [{"role": "system", "content": prompt_instructions.strip()}]
    for entry in st.session_state.history:
        messages.append(entry)
    messages.insert(1, {
        "role": "system",
        "content": f"Use the following Ayurvedic knowledge base for reference only. Do not quote or reference directly:\n\n{context}"
    })

    # Call Groq
    try:
        response = requests.post(
            GROQ_API_URL,
            headers=HEADERS,
            json={"model": "llama3-8b-8192", "messages": messages}
        )
        reply = response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        reply = f"❌ Error: {e}"

    # Store assistant reply
    st.session_state.history.append({"role": "assistant", "content": reply})

# Display conversation
for entry in st.session_state.history:
    if entry["role"] == "user":
        st.chat_message("user").markdown(entry["content"])
    else:
        st.chat_message("assistant").markdown(entry["content"])
