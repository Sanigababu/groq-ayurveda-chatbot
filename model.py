import os
import json
import sys
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Setup
DATA_DIR = "data"
chunks = []

# Optional CLI flag: --force
FORCE_REBUILD = "--force" in sys.argv

# Load and parse all JSON and JSONL files
for filename in os.listdir(DATA_DIR):
    file_path = os.path.join(DATA_DIR, filename)

    if filename.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for entry in data:
                if isinstance(entry, dict) and "content" in entry:
                    chunks.append(entry["content"])
                elif isinstance(entry, str):
                    chunks.append(entry)

    elif filename.endswith(".jsonl"):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if isinstance(entry, dict) and "content" in entry:
                        chunks.append(entry["content"])
                    elif isinstance(entry, str):
                        chunks.append(entry)
                except json.JSONDecodeError:
                    continue

# Load model
print("🔄 Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ChromaDB client and collection
print("🧠 Connecting to ChromaDB...")
client = chromadb.PersistentClient(path="./chroma_store")
collection = client.get_or_create_collection("chat_chunks")

# If forced, delete existing collection content
if FORCE_REBUILD:
    print("⚠️  Force rebuild enabled: Deleting existing documents...")
    collection.delete()

# Check existing documents
existing = collection.count()
if existing > 0 and not FORCE_REBUILD:
    print(f"✅ Skipping embedding — {existing} documents already in collection.")
    sys.exit(0)

# Embed
print(f"🔄 Encoding {len(chunks)} chunks...")
embeddings = model.encode(chunks, batch_size=32, show_progress_bar=True)

# Store in ChromaDB
print("💾 Storing in ChromaDB...")
for i in range(0, len(chunks), 50):
    batch_chunks = chunks[i:i+50]
    batch_embeddings = embeddings[i:i+50]
    ids = [f"doc_{i+j}" for j in range(len(batch_chunks))]
    collection.add(documents=batch_chunks, embeddings=batch_embeddings.tolist(), ids=ids)

print("✅ Done storing embeddings.")
