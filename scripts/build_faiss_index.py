import faiss
import numpy as np
import json

def build_faiss_index(data_file, index_file):
    # Load embeddings
    with open(data_file, "r") as f:
        data = json.load(f)

    embeddings = np.array([item["embedding"] for item in data], dtype="float32")
    
    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 similarity
    index.add(embeddings)
    
    # Save index to a file
    faiss.write_index(index, index_file)
    print(f"FAISS index saved to {index_file}")

if __name__ == "__main__":
    build_faiss_index("prepared_data.json", "faiss_index.bin")
