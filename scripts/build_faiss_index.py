import faiss
import numpy as np
import json

def build_faiss_index(data_file, index_file):
    """
    Build and save a FAISS index from the given data file, using cosine similarity.

    Args:
        data_file (str): Path to the JSON file containing embeddings.
        index_file (str): Path to save the FAISS index.
    """
    # Load embeddings
    with open(data_file, "r") as f:
        data = json.load(f)

    embeddings = np.array([item["embedding"] for item in data], dtype="float32")
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)

    # Create FAISS index
    index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner Product (IP) for cosine similarity
    index.add(embeddings)
    
    # Save index to a file
    faiss.write_index(index, index_file)
    print(f"FAISS index saved to {index_file}")

if __name__ == "__main__":
    # Build FAISS index for pretrained embeddings
    build_faiss_index(
        data_file="prepared_data_pretrained.json",
        index_file="faiss_index_pretrained.bin"
    )
    
    # Build FAISS index for fine-tuned embeddings
    build_faiss_index(
        data_file="prepared_data_finetuned.json",
        index_file="faiss_index_finetuned.bin"
    )
