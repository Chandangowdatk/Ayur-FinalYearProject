import json
import numpy as np
import uuid
import sqlite3
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility

# Step 1: Load the JSON File Containing Sanskrit Embeddings
file_path = r'/Users/chandangowdatk/Development/Ayur-FinalYearProject/MINILM-embed-for-Scientific_Basis_for_Ayurvedic_Therapies.txt'  # Update with actual path
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

sentences = []
embeddings = []
for key, value in data.items():
    sentences.append(value["sentence"])
    embeddings.append(value["embedding"])

# Step 2: Generate unique IDs for sentences
sentence_ids = [str(uuid.uuid4())[:36] for _ in range(len(sentences))]

# Step 3: Store the full sentences in SQLite
sqlite_db_path = "L2_minilm_sentences.db"
conn = sqlite3.connect(sqlite_db_path)
cursor = conn.cursor()

# Create table if it doesn't exist
cursor.execute('''
CREATE TABLE IF NOT EXISTS sentences (
    id TEXT PRIMARY KEY,
    full_text TEXT
)
''')

# Clear existing data if any
cursor.execute("DELETE FROM sentences")

# Insert sentences with their IDs - in batches to avoid SQLite limitations
BATCH_SIZE = 1000
for i in range(0, len(sentences), BATCH_SIZE):
    batch_data = [(sentence_ids[j], sentences[j]) for j in range(i, min(i + BATCH_SIZE, len(sentences)))]
    cursor.executemany("INSERT INTO sentences (id, full_text) VALUES (?, ?)", batch_data)
    conn.commit()
    print(f"✅ Stored batch {i//BATCH_SIZE + 1}/{(len(sentences)-1)//BATCH_SIZE + 1} in SQLite")

print(f"✅ Successfully stored all {len(sentences)} full sentences in SQLite database!")

# Step 4: Connect to Milvus
connections.connect(host="localhost", port="19530")  # Update if using cloud

# Step 5: Define and create Milvus Collection
collection_name = "L2_minilm_rag"

# Check if collection exists and drop it if it does
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)
    print(f"🗑 Dropped existing collection: {collection_name}")

# Create new collection with sentence_id instead of full sentence
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="sentence_id", dtype=DataType.VARCHAR, max_length=36),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=len(embeddings[0]))
]
schema = CollectionSchema(fields, description="English sentence embeddings")
milvus_collection = Collection(name=collection_name, schema=schema)
print(f"✅ Created new collection: {collection_name}")

# Step 6: Insert in smaller batches to avoid gRPC message size limitation
BATCH_SIZE = 5000  # Adjust based on your embedding dimensions
total_batches = (len(embeddings) - 1) // BATCH_SIZE + 1

for i in range(0, len(embeddings), BATCH_SIZE):
    end_idx = min(i + BATCH_SIZE, len(embeddings))
    batch_ids = sentence_ids[i:end_idx]
    batch_embeddings = embeddings[i:end_idx]
    
    data_to_insert = [batch_ids, batch_embeddings]
    
    insert_result = milvus_collection.insert(data_to_insert)
    milvus_collection.flush()
    
    print(f"✅ Inserted batch {i//BATCH_SIZE + 1}/{total_batches} into Milvus ({end_idx - i} records)")

print(f"✅ Successfully inserted all {len(sentence_ids)} embeddings into Milvus!")

# Step 7: Create index for faster search
index_params = {
    "metric_type": "L2",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128}
}
milvus_collection.create_index("embedding", index_params)
print("✅ Created index on embeddings!")

# Step 8: Verify Data Storage
milvus_collection.load()
print(f"📊 Total embeddings stored in Milvus: {milvus_collection.num_entities}")

# Step 9: Example search function
def search_similar_sanskrit(query_embedding, top_k=5):
    """
    Search for similar Sanskrit sentences using a query embedding.
    Returns the full sentences from SQLite.
    """
    # Search in Milvus
    search_results = milvus_collection.search(
        data=[query_embedding],  
        anns_field="embedding",  
        param={"metric_type": "L2", "params": {"nprobe": 10}},  
        limit=top_k,  
        output_fields=["sentence_id"]
    )
    
    # Process results
    results = []
    for hit in search_results[0]:
        sentence_id = hit.entity.get('sentence_id')
        
        # Get full text from SQLite
        cursor.execute("SELECT full_text FROM sentences WHERE id = ?", (sentence_id,))
        result = cursor.fetchone()
        
        if result:
            results.append({
                "id": sentence_id,
                "distance": hit.distance,
                "full_text": result[0]
            })
    
    return results

# Example usage with a sample query (use a real embedding when actually querying)
print("\n🔍 To search, use the search_similar_sanskrit function with your query embedding:")
print("example_results = search_similar_sanskrit(your_query_embedding, top_k=5)")

# Cleanup connections before exiting
connections.disconnect("default")
conn.close()
print("\n✅ Connections closed.")








# from groq import Groq
# import json
# import sqlite3
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from pymilvus import connections, Collection

# # 📌 Constants
# MILVUS_HOST = "localhost"
# MILVUS_PORT = "19530"
# MILVUS_COLLECTION = "L2_minilm_rag"
# SQLITE_DB_PATH = "L2_minilm_sentences.db"
# GROQ_API_KEY = "Your_API_Key"  # 🔹 Replace with your actual API key

# # ✅ Initialize Groq client
# client = Groq(api_key=GROQ_API_KEY)

# # ✅ Load the embedding model (using all-MiniLM-L6-v2)
# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# # ✅ Connect to Milvus
# connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
# milvus_collection = Collection(name=MILVUS_COLLECTION)

# # ✅ Connect to SQLite
# conn = sqlite3.connect(SQLITE_DB_PATH)
# cursor = conn.cursor()

# def query_similar_sanskrit(english_query, top_k=40):
#     """Query Milvus for similar Sanskrit sentences using an English query."""
#     query_embedding = model.encode([english_query])[0].tolist()

#     search_results = milvus_collection.search(
#         data=[query_embedding],
#         anns_field="embedding",
#         param={"metric_type": "L2", "params": {"nprobe": 10}},
#         limit=top_k,
#         output_fields=["sentence_id"]
#     )

#     results = []
#     for hit in search_results[0]:
#         sentence_id = hit.entity.get("sentence_id")

#         cursor.execute("SELECT full_text FROM sentences WHERE id = ?", (sentence_id,))
#         result = cursor.fetchone()

#         if result:
#             results.append(result[0])  # Only store the sentence text

#     return results

# def generate_response(question, context_sentences):
#     """Use Groq's LLaMA API to generate an answer based on the retrieved Sanskrit sentences."""
#     context_text = "\n".join(context_sentences)
#     prompt = (
#         "You are an Ayurvedic doctor. Answer the question using Ayurvedic principles. "
#         "Replace common medical terms with Ayurvedic terminology if appropriate. "
#         "Ensure the response aligns with Ayurveda's holistic approach."
#         "Answer this question by only considering the retrieved context and don't generate answers of your own."
#         "If contexts are not enough please response as sorry no data available.\n\n"
#         f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"
#     )

#     completion = client.chat.completions.create(
#         model="llama-3.2-11b-vision-preview",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=1,
#         max_tokens=1024,
#         top_p=1,
#         stream=False,
#     )

#     return completion.choices[0].message.content.strip()

# # ✅ Test the function
# test_query = "What are the therapies for back pain?"
# top_k_results = 40
# retrieved_sentences = query_similar_sanskrit(test_query,top_k_results)
# # print(retrieved_sentences)

# if retrieved_sentences:
#     response = generate_response(test_query, retrieved_sentences)
#     print("\n🔍 Query:", test_query)
#     print("🔹 Answer:\n", response)
# else:
#     print("No relevant Answer found.")

# # ✅ Close connections
# connections.disconnect("default")
# conn.close()