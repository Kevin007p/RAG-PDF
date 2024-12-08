import faiss
import numpy as np
import json
import logging
logging.basicConfig(level=logging.INFO)

from sentence_transformers import SentenceTransformer
logging.info("SentenceTransformer imported successfully.")

def query_faiss_index(query, index_file, data_file, top_k=5):
    # Load FAISS index
    index = faiss.read_index(index_file)

    # Load data
    with open(data_file, "r") as f:
        data = json.load(f)
    texts = [item["content"] for item in data]

    # Encode the query
    model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
    logging.info("Model loaded successfully.")
    query_embedding = model.encode(query).astype("float32")

    # Perform search
    distances, indices = index.search(np.array([query_embedding]), top_k)

    # Print results
    print(f"Top {top_k} results for query: '{query}'")
    for dist, idx in zip(distances[0], indices[0]):
        print(f"Distance: {dist}, Content: {texts[idx]}")

if __name__ == "__main__":
    query = input("Enter your query: ")
    query_faiss_index(query, "faiss_index.bin", "prepared_data.json")
