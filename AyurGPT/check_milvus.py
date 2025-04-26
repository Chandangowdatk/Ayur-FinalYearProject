from pymilvus import connections, Collection, utility

# Connect to Milvus
connections.connect(host="127.0.0.1", port=19530)

# Check if collection exists
collection_name = "L2_minilm_rag"
print(f"Does collection '{collection_name}' exist? {utility.has_collection(collection_name)}")

if utility.has_collection(collection_name):
    # Load the collection
    collection = Collection(collection_name)
    
    # Get collection statistics
    print(f"Collection schema: {collection.schema}")
    print(f"Collection description: {collection.description}")
    
    # Get entity count
    print(f"Number of entities: {collection.num_entities}")
    
    # Get index details
    print(f"Index information: {collection.index().params}")
    
    # Sample a few vectors if possible
    if collection.num_entities > 0:
        try:
            results = collection.query(expr="id >= 0", output_fields=["sentence_id"], limit=5)
            print(f"Sample query results: {results}")
        except Exception as e:
            print(f"Error querying collection: {e}")
    
    # List all collections
    all_collections = utility.list_collections()
    print(f"All collections: {all_collections}")
else:
    print("Collection does not exist!") 