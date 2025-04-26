from groq import Groq
import json
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection

# 📌 Constants
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
MILVUS_COLLECTION = "L2_minilm_rag"
SQLITE_DB_PATH = "L2_minilm_sentences.db"
GROQ_API_KEY = "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"  # 🔹 Replace with your actual API key

# ✅ Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# ✅ Load the embedding model (using all-MiniLM-L6-v2)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ✅ Connect to Milvus
connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
milvus_collection = Collection(name=MILVUS_COLLECTION)

# ✅ Connect to SQLite
conn = sqlite3.connect(SQLITE_DB_PATH)
cursor = conn.cursor()

def query_similar_sanskrit(english_query, top_k=40):
    """Query Milvus for similar Sanskrit sentences using an English query."""
    query_embedding = model.encode([english_query])[0].tolist()

    search_results = milvus_collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["sentence_id"]
    )

    results = []
    for hit in search_results[0]:
        sentence_id = hit.entity.get("sentence_id")

        cursor.execute("SELECT full_text FROM sentences WHERE id = ?", (sentence_id,))
        result = cursor.fetchone()

        if result:
            results.append(result[0])  # Only store the sentence text

    return results

def generate_response(question, context_sentences):
    """Use Groq's LLaMA API to generate an answer based on the retrieved Sanskrit sentences."""
    context_text = "\n".join(context_sentences)
    prompt = (
        "You are an Ayurvedic doctor. Answer the question using Ayurvedic principles. "
        "Replace common medical terms with Ayurvedic terminology if appropriate. "
        "Ensure the response aligns with Ayurveda's holistic approach."
        "Answer this question by only considering the retrieved context and don't generate answers of your own."
        "If contexts are not enough please response as sorry no data available."
        "If available in context, segregate the answers as overview, home remedies, scientific studies, recommendations."
        "Provide more elaborative answers.\n\n"
        f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"
    )

    completion = client.chat.completions.create(
        model="llama-3.2-11b-vision-preview",
        messages=[{"role": "user", "content": prompt}],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=False,
    )

    return completion.choices[0].message.content.strip()

# ✅ Test the function
# test_query = "What are the therapies for back pain?"
test_query = input("Enter the question : \n")
top_k_results = 40
retrieved_sentences = query_similar_sanskrit(test_query,top_k_results)
print(retrieved_sentences)

if retrieved_sentences:
    response = generate_response(test_query, retrieved_sentences)
    print("\n🔍 Query:", test_query)
    print("🔹 Answer:\n", response)
else:
    print("No relevant Answer found.")

# ✅ Close connections
connections.disconnect("default")
conn.close()