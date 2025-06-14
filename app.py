import os
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import chromadb
import sys
import types

# Prevent Streamlit from inspecting torch.classes
import torch
torch.classes = types.SimpleNamespace()

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

# System Prompt for the Ayurvedic Assistant
prompt_instructions = """
You are an Ayurvedic Assistant, an expert in traditional Indian medicine and holistic wellness.

Guidelines:
- Focus only on Ayurvedic health-related advice.
- Start with a warm, brief greeting, but don’t repeat it with each query.
- Ask follow-up or clarifying questions before giving advice.
- Keep responses short (max one paragraph or 3–4 sentences).
- Be conversational, natural, and engaging like ChatGPT.
- Base your advice on Ayurvedic principles—use natural remedies, herbs, dietary/lifestyle changes.
- If appropriate, briefly explain relevant Ayurvedic concepts like doshas (Vata, Pitta, Kapha), dhatus, etc.
- Use Sanskrit terms only when helpful and always explain them.
- Be warm, compassionate, and respectful.
- Never give advice without knowing enough symptoms.
- Emphasize this is not a replacement for professional medical care.
"""

# Load embedding model and ChromaDB
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_store")
collection = client.get_or_create_collection("chat_chunks")

# Function to retrieve documents
def search_docs(query: str, top_k: int = 3, max_doc_chars: int = 300):
    query_embedding = model.encode([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )
    documents = results.get("documents", [[]])[0]
    return [doc[:max_doc_chars] for doc in documents if doc]

# Streamlit setup
st.set_page_config(page_title="Ayurvedic Assistant", page_icon="🧘‍♀️")
st.title("🌿 Ayurvedic Assistant Chatbot")

# Session state to hold chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": prompt_instructions.strip()}]

# Display previous conversation
for msg in st.session_state.messages[1:]:  # Skip system prompt
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Ask something related to Ayurveda...")

if user_input:
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Search for document context
    with st.spinner("🔍 Searching knowledge base..."):
        docs = search_docs(user_input, top_k=3)
        context = "\n".join(docs)[:3000]  # Keep it well under Groq token limit

    # Create new message list just for this API call
    temp_messages = st.session_state.messages.copy()
    temp_messages.insert(1, {
        "role": "system",
        "content": f"Use the following Ayurvedic knowledge base for reference only. Do not quote or reference directly:\n\n{context}"
    })

    # Generate model response
    with st.spinner("💬 Generating response..."):
        response = requests.post(
            GROQ_API_URL,
            headers=HEADERS,
            json={"model": "llama3-8b-8192", "messages": temp_messages}
        )

        if response.status_code == 200:
            reply = response.json()["choices"][0]["message"]["content"]
            st.session_state.messages.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)
        else:
            error_msg = f"Error: {response.text}"
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            with st.chat_message("assistant"):
                st.error(error_msg)
