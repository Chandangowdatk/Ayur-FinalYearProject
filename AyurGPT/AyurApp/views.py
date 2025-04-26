import time
import os
import json
import sqlite3
# import numpy as np
from django.http import JsonResponse
from rest_framework.decorators import api_view
from groq import Groq
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection
from dotenv import load_dotenv
from pathlib import Path

# 📌 Constants
load_dotenv()
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = 19530
MILVUS_COLLECTION = "ayur_fixed_rag"

# Get the base directory and create absolute path for SQLite database
BASE_DIR = Path(__file__).resolve().parent.parent
SQLITE_DB_PATH = os.path.join(BASE_DIR, "L2_minilm_sentences.db")
print(f"Using SQLite database at: {SQLITE_DB_PATH}")

# GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # 🔹 Replace with your actual API key
# GROQ_API_KEY="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"
GROQ_API_KEY="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"

# ✅ Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# ✅ Load the embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ✅ Connect to Milvus
connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
milvus_collection = Collection(name=MILVUS_COLLECTION)

# ✅ Connect to SQLite
conn = sqlite3.connect(SQLITE_DB_PATH)
cursor = conn.cursor()

def query_similar_sanskrit(english_query, top_k=40):
    """Query Milvus for similar Sanskrit sentences using an English query."""
    
    # Open a new SQLite connection for this request
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()

    query_embedding = model.encode([english_query])[0].tolist()
    
    print(f"Querying Milvus with embedding of size: {len(query_embedding)}")

    search_results = milvus_collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["sentence_id"]
    )
    
    print(f"Milvus returned {len(search_results[0])} results")

    results = []
    for hit in search_results[0]:
        sentence_id = hit.entity.get("sentence_id")
        distance = hit.distance
        print(f"Found hit with ID {sentence_id}, distance: {distance}")
        cursor.execute("SELECT full_text FROM sentences WHERE id = ?", (sentence_id,))
        result = cursor.fetchone()
        if result:
            results.append(result[0])
        else:
            print(f"No text found for sentence_id: {sentence_id}")

    # Close the connection to avoid thread issues
    conn.close()
    
    print(f"Retrieved {len(results)} Sanskrit sentences")
    return results


def generate_response(question, context_sentences):
    """Use Groq's LLaMA API to generate an answer based on the retrieved Sanskrit sentences."""
    context_text = "\n".join(context_sentences)
    prompt = (
        "You are an expert Ayurvedic doctor. Answer the question using the provided context.\n\n"
        "Guidelines:\n"
        "- Extract as much relevant information as possible from the context.\n"
        "- Only if you absolutely cannot find ANY relevant information, respond with 'I don't have enough information about this topic.'\n"
        "- Otherwise, do your best to answer with the available information.\n"
        "- Replace common medical terms with Ayurvedic terminology if appropriate.\n"
        "- Ensure the response aligns with Ayurveda's holistic approach.\n"
        "- If enough information is available, organize your answer into: Overview, Home Remedies, Dietary Recommendations, and Scientific Studies.\n"
        "- Be thorough and detailed in your response.\n\n"
        f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"
    )

    print(f"Sending prompt of length {len(prompt)} to Groq")
    
    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1024,
        top_p=1,
        stream=False,
    )

    return completion.choices[0].message.content.strip()

@api_view(["POST"])
def chat(request):
    """API endpoint to handle user queries."""
    try:
        data = json.loads(request.body)
        question = data.get("question", "")
        print(f"Received question: {question}")
        
        if not question:
            return JsonResponse({"error": "No question provided"}, status=400)

        retrieved_sentences = query_similar_sanskrit(question, top_k=40)
        
        if retrieved_sentences:
            response = generate_response(question, retrieved_sentences)
        else:
            response = "I don't have enough information to answer this question."
        
        print(f"Number of retrieved sentences: {len(retrieved_sentences)}")
        print(f"First few retrieved sentences: {retrieved_sentences[:2]}")
        print(f"Response: {response[:100]}...")

        return JsonResponse({"question": question, "answer": response})

    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return JsonResponse({"error": str(e)}, status=500)