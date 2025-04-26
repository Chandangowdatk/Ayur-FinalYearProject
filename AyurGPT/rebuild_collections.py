import os
import sqlite3
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

# Constants
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = 19530
NEW_MILVUS_COLLECTION = "ayur_fixed_rag"
BASE_DIR = Path(__file__).resolve().parent
SQLITE_DB_PATH = os.path.join(BASE_DIR, "L2_minilm_sentences.db")
print(f"Using SQLite database at: {SQLITE_DB_PATH}")

# Initialize model
print("Loading embedding model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embedding_dim = model.get_sentence_embedding_dimension()
print(f"Model loaded with embedding dimension: {embedding_dim}")

# Connect to Milvus
connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

# Create the schema for the new collection
print("Creating new collection schema...")
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="sentence_id", dtype=DataType.VARCHAR, max_length=36),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim)
]
schema = CollectionSchema(fields=fields, description="Ayurvedic text embeddings with fixed IDs")

# Drop existing collection if it exists
if utility.has_collection(NEW_MILVUS_COLLECTION):
    utility.drop_collection(NEW_MILVUS_COLLECTION)
    print(f"Dropped existing collection: {NEW_MILVUS_COLLECTION}")

# Create the new collection
collection = Collection(name=NEW_MILVUS_COLLECTION, schema=schema)
print(f"Created new collection: {NEW_MILVUS_COLLECTION}")

# Connect to SQLite
conn = sqlite3.connect(SQLITE_DB_PATH)
cursor = conn.cursor()

# Get the total number of rows
cursor.execute("SELECT COUNT(*) FROM sentences")
total_rows = cursor.fetchone()[0]
print(f"Total rows in SQLite: {total_rows}")

# Process in batches
batch_size = 100
num_batches = (total_rows + batch_size - 1) // batch_size

for batch_num in tqdm(range(num_batches), desc="Processing batches"):
    offset = batch_num * batch_size
    
    # Get a batch of text from SQLite
    cursor.execute(f"SELECT id, full_text FROM sentences LIMIT {batch_size} OFFSET {offset}")
    batch_data = cursor.fetchall()
    
    if not batch_data:
        continue
    
    # Prepare the data for insertion
    sentence_ids = []
    texts = []
    
    for sentence_id, text in batch_data:
        sentence_ids.append(sentence_id)
        texts.append(text)
    
    # Generate embeddings
    embeddings = model.encode(texts).tolist()
    
    # Insert into Milvus
    entities = [
        sentence_ids,
        embeddings
    ]
    
    collection.insert(entities)

print("Creating index...")
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128}
}
collection.create_index(field_name="embedding", index_params=index_params)
print("Index created")

print("Loading collection into memory...")
collection.load()

# Test a query
print("Testing with a sample query...")
query_text = "Ayurvedic treatments for diabetes"
query_embedding = model.encode([query_text])[0].tolist()

search_results = collection.search(
    data=[query_embedding],
    anns_field="embedding",
    param={"metric_type": "L2", "params": {"nprobe": 10}},
    limit=5,
    output_fields=["sentence_id"]
)

print("Sample search results:")
for i, hit in enumerate(search_results[0]):
    sentence_id = hit.entity.get("sentence_id")
    distance = hit.distance
    
    cursor.execute("SELECT substr(full_text, 1, 100) FROM sentences WHERE id = ?", (sentence_id,))
    result = cursor.fetchone()
    text_preview = result[0] if result else "NOT FOUND"
    
    print(f"Hit #{i+1}: ID {sentence_id}, distance: {distance}")
    print(f"   Text: {text_preview}...")

# Close connections
conn.close()
print("\nDone! To use the new collection, update MILVUS_COLLECTION in your views.py to 'ayur_fixed_rag'") 