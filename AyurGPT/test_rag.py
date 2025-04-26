import os
import sqlite3
from pathlib import Path
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection

# Constants (same as in views.py)
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = 19530
MILVUS_COLLECTION = "ayur_fixed_rag"  # Use the fixed collection
BASE_DIR = Path(__file__).resolve().parent
SQLITE_DB_PATH = os.path.join(BASE_DIR, "L2_minilm_sentences.db")
print(f"Using SQLite database at: {SQLITE_DB_PATH}")

# Initialize model and connections
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
milvus_collection = Collection(name=MILVUS_COLLECTION)
milvus_collection.load()

def query_similar_sanskrit(english_query, top_k=5):
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
    for i, hit in enumerate(search_results[0]):
        sentence_id = hit.entity.get("sentence_id")
        distance = hit.distance
        print(f"Hit #{i+1}: ID {sentence_id}, distance: {distance}")
        cursor.execute("SELECT full_text FROM sentences WHERE id = ?", (sentence_id,))
        result = cursor.fetchone()
        if result:
            text = result[0]
            print(f"   Text: {text[:100]}..." if len(text) > 100 else f"   Text: {text}")
            results.append(text)
        else:
            print(f"   No text found for sentence_id: {sentence_id}")

    # Close the connection to avoid thread issues
    conn.close()
    
    print(f"Retrieved {len(results)} Sanskrit sentences")
    return results

# Test with some sample queries
test_queries = [
    "What are Ayurvedic treatments for diabetes?",
    "How does Ayurveda treat skin conditions?",
    "What foods should I eat for Pitta dosha?",
    "Ayurvedic herbs for stress and anxiety"
]

for query in test_queries:
    print("\n" + "="*80)
    print(f"TESTING QUERY: {query}")
    print("="*80)
    results = query_similar_sanskrit(query)
    print(f"Summary: Retrieved {len(results)} results for query '{query}'")
    print("="*80) 