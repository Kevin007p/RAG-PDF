import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

class RAGModel:
    def __init__(self, index_file, data_file, embedding_model="multi-qa-mpnet-base-dot-v1"):
        self.index = faiss.read_index(index_file)
        with open(data_file, "r") as f:
            self.data = json.load(f)
        self.embedding_model = SentenceTransformer(embedding_model, device='mps')
        self.generator = pipeline("text2text-generation", model="google/flan-t5-base", device=0)

    def retrieve(self, query, top_k=5):
        query_embedding = self.embedding_model.encode(query).astype("float32")
        distances, indices = self.index.search(np.array([query_embedding]), top_k)
        results = [{"content": self.data[idx]["content"], "distance": dist} for dist, idx in zip(distances[0], indices[0])]
        return results

    def generate_answer(self, query, top_k=5):
        retrieved_docs = self.retrieve(query, top_k)
        context = " ".join([doc["content"] for doc in retrieved_docs[:top_k]])

        prompt = (
            f"Question: {query}\n"
            f"Relevant Context: {context}\n"
            f"Explain in detail and avoid repetition. Provide an extensive answer of at least 50 words."
        )

        answer = self.generator(prompt, max_length=150, min_length=50, num_return_sequences=1)

        return {"query": query, "answer": answer[0]["generated_text"], "retrieved_docs": retrieved_docs}

if __name__ == "__main__":
    rag_model = RAGModel(index_file="faiss_index.bin", data_file="prepared_data.json")
    query = input("Enter your query: ")
    result = rag_model.generate_answer(query)
    print(f"Query: {result['query']}")
    print(f"Answer: {result['answer']}")
    print("\nRetrieved Documents:")
    for doc in result["retrieved_docs"]:
        print(f"- {doc['content']} (Distance: {doc['distance']:.2f})")
