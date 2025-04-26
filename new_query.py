import os
import json
import sqlite3
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection

# ✅ Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ Error: GROQ_API_KEY is missing. Set it in your environment variables.")

# 📌 Constants
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
MILVUS_COLLECTION = "L2_minilm_rag"
SQLITE_DB_PATH = "L2_minilm_sentences.db"

# ✅ Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# ✅ Load embedding model (using all-MiniLM-L6-v2)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ✅ Connect to Milvus with alias
connections.connect(host=MILVUS_HOST, port=MILVUS_PORT, alias="default")
milvus_collection = Collection(name=MILVUS_COLLECTION)

# ✅ Connect to SQLite
conn = sqlite3.connect(SQLITE_DB_PATH)
cursor = conn.cursor()

def query_similar_sanskrit(english_query, top_k=40):
    """Query Milvus for similar Sanskrit sentences using an English query."""
    try:
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
            if sentence_id is None:
                continue  # Skip if sentence_id is missing

            cursor.execute("SELECT full_text FROM sentences WHERE id = ?", (sentence_id,))
            result = cursor.fetchone()

            if result:
                results.append(result[0])  # Store only the sentence text

        return results
    except Exception as e:
        print(f"❌ Error in query_similar_sanskrit: {e}")
        return []

def generate_response(question, context_sentences):
    """Use Groq's LLaMA API to generate an answer based on the retrieved Sanskrit sentences."""
    if not context_sentences:
        return "❌ No relevant data found in the context."

    try:
        context_text = "\n".join(context_sentences)
        prompt = (
            "You are an Ayurvedic doctor. Answer the question using Ayurvedic principles. "
            "Replace common medical terms with Ayurvedic terminology if appropriate. "
            "Ensure the response aligns with Ayurveda's holistic approach.\n\n"
            "📝 Instructions:\n"
            "1. Answer strictly based on the retrieved context.\n"
            "2. If the context is insufficient, respond with: 'Sorry, no data available.'\n"
            "3. If relevant, categorize the response under:\n"
            "   - **Overview**\n"
            "   - **Home Remedies**\n"
            "   - **Scientific Studies**\n"
            "   - **Recommendations**\n"
            "4. Provide detailed, elaborative answers.\n\n"
            f"🔍 Context:\n{context_text}\n\n🔹 Question: {question}\n🔹 Answer:"
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
    except Exception as e:
        print(f"❌ Error in generate_response: {e}")
        return "❌ Unable to generate response."

# ✅ Continuous Loop for Multiple Queries
if __name__ == "__main__":
    print("\n💬 Ask your questions about Ayurveda (Type 'exit' or 'stop' to stop)\n")

    while True:
        test_query = input("🔹 Enter your question: ").strip()

        # Exit Condition
        if test_query.lower() in ["exit", "quit", "stop"]:
            print("🚪 Exiting... Dhanyavaadah! 🙏")
            break

        top_k_results = 40
        retrieved_sentences = query_similar_sanskrit(test_query, top_k_results)
        # print(retrieved_sentences)

        if retrieved_sentences:
            response = generate_response(test_query, retrieved_sentences)
            print("\n🔍 Query:", test_query)
            print("🔹 Answer:\n", response)
        else:
            print("❌ No relevant answer found.")

# ✅ Close connections after exiting
connections.disconnect(alias="default")
conn.close()
