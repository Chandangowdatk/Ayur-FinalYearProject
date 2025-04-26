from sentence_transformers import SentenceTransformer
import json

# Initialize the SentenceTransformer model with all-MiniLM-L6-v2
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def chunk_by_sentence(text):
    """
    Splits the text into chunks based on sentence boundaries ('।' for Sanskrit).
    
    :param text: Cleaned text to be split into sentences.
    :return: A list of sentences extracted from the text.
    """
    sentences = []
    tmp_sentence = ""
    for char in text:
        if char in [".", "!", "?"]:
            if tmp_sentence.strip():
                sentences.append(tmp_sentence.strip())
            tmp_sentence = ""
        else:
            tmp_sentence += char
    # Add any remaining text as the last sentence
    if tmp_sentence.strip():
        sentences.append(tmp_sentence.strip())
    return sentences

def generate_embeddings(text):
    sentences = chunk_by_sentence(text)
    embeddings = model.encode(sentences)
    return sentences, embeddings

# Load Sanskrit text from the input file
input_file_path = r'\Users\srivatsa gubbi\OneDrive\Desktop\AyurGPT\Milvus trial\Scientific_Basis_for_Ayurvedic_Therapies.txt'
with open(input_file_path, 'r', encoding='utf-8') as file:
    sanskrit_text = file.read()

# Generate embeddings
sentences, embeddings = generate_embeddings(sanskrit_text)

# Prepare data for saving
embeddings_dict = {idx: {"sentence": sentence, "embedding": embedding.tolist()} 
                    for idx, (sentence, embedding) in enumerate(zip(sentences, embeddings))}

# Save embeddings to an output file
output_file_path = r'\Users\srivatsa gubbi\OneDrive\Desktop\AyurGPT\Milvus trial\MINILM-embed-for-Scientific_Basis_for_Ayurvedic_Therapies.txt'
with open(output_file_path, 'w', encoding='utf-8') as file:
    json.dump(embeddings_dict, file, ensure_ascii=False, indent=4)

print(f"Embeddings saved to {output_file_path}")