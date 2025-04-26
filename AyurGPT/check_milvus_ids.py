from pymilvus import connections, Collection, utility
import sqlite3
import os
from pathlib import Path

# Constants
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = 19530
MILVUS_COLLECTION = "L2_minilm_rag"
BASE_DIR = Path(__file__).resolve().parent
SQLITE_DB_PATH = os.path.join(BASE_DIR, "L2_minilm_sentences.db")
print(f"Using SQLite database at: {SQLITE_DB_PATH}")

# Connect to Milvus
connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
milvus_collection = Collection(name=MILVUS_COLLECTION)
milvus_collection.load()

# Connect to SQLite
conn = sqlite3.connect(SQLITE_DB_PATH)
cursor = conn.cursor()

# Get sample IDs from SQLite
cursor.execute("SELECT id FROM sentences LIMIT 10")
sqlite_ids = [row[0] for row in cursor.fetchall()]
print(f"Sample SQLite IDs: {sqlite_ids}")

# Query Milvus to get some IDs
results = milvus_collection.query(expr="id >= 0", output_fields=["sentence_id"], limit=10)
milvus_ids = [result["sentence_id"] for result in results]
print(f"Sample Milvus IDs: {milvus_ids}")

# Check if any of the Milvus IDs exist in SQLite
placeholders = ','.join(['?'] * len(milvus_ids))
cursor.execute(f"SELECT id, substr(full_text, 1, 100) FROM sentences WHERE id IN ({placeholders})", milvus_ids)
matches = cursor.fetchall()
print(f"Matching IDs in SQLite: {matches}")

# Calculate statistics
total_milvus_entities = milvus_collection.num_entities
cursor.execute("SELECT COUNT(*) FROM sentences")
total_sqlite_rows = cursor.fetchone()[0]

print(f"Total Milvus entities: {total_milvus_entities}")
print(f"Total SQLite rows: {total_sqlite_rows}")

# Check if we need to sync the IDs
print("\nAnalyzing mismatch...\n")

# Get Milvus sentence_id format
print("Getting sample sentence_id format from Milvus...")
sample_results = milvus_collection.query(expr="id >= 0", output_fields=["sentence_id"], limit=5)
for i, result in enumerate(sample_results):
    print(f"Milvus sentence_id format example {i+1}: {result['sentence_id']}")

# Close connections
conn.close() 