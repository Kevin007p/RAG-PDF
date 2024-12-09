import faiss
import json
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from transformers import pipeline

class RAGModel:
    def __init__(self, index_file, data_file, embedding_model):
        """
        Initialize the RAG model with a FAISS index, data file, and embedding model.

        Args:
            index_file (str): Path to the FAISS index file.
            data_file (str): Path to the data file (JSON).
            embedding_model (str): Path to the SentenceTransformer model.
        """
        self.index = faiss.read_index(index_file)
        with open(data_file, "r") as f:
            self.data = json.load(f)
        self.embedding_model = SentenceTransformer(embedding_model, device='mps')
        self.generator = pipeline("text2text-generation", model="google/flan-t5-base", device=0)

    def clean_text(self, text):
        """
        Clean text by removing unwanted characters, math symbols, and excessive whitespace.
        """
        text = re.sub(r"[^\w\s.,!?-]", "", text)  # Remove raw LaTeX/math symbols and special characters
        text = re.sub(r"\s+", " ", text)  # Normalize whitespace
        return text.strip()

    def refine_answer(self, answer):
        """
        Post-process the generated answer to remove repetitions or noisy outputs.
        """
        # Remove repeated phrases
        answer = re.sub(r"(.*)\.\s+\1", r"\1.", answer)
        # Clean up stray characters
        answer = re.sub(r"[^\w\s.,!?-]", "", answer)
        return answer.strip()

    def retrieve(self, query, top_k=5):
        """
        Retrieve the top-k relevant documents for a given query.
        """
        query_embedding = self.embedding_model.encode(query).astype("float32")
        distances, indices = self.index.search(np.array([query_embedding]), top_k)
        results = [{"content": self.data[idx]["content"], "distance": dist} for dist, idx in zip(distances[0], indices[0])]
        return results

    def generate_answer(self, query, top_k=5):
        """
        Generate an answer to the query using retrieved documents and a pre-trained text generator.
        """
        retrieved_docs = self.retrieve(query, top_k)

        # Filter and clean context
        relevant_docs = [doc for doc in retrieved_docs if "hidden" in doc["content"].lower()]
        if not relevant_docs:  # Fall back to top-k if no matches
            relevant_docs = retrieved_docs[:top_k]

        context = " ".join([self.clean_text(doc["content"]) for doc in relevant_docs])

        # Summarize if context is too long
        if len(context.split()) > 512:  # Assuming 512 tokens as model's limit
            context = self.generator(
                f"Summarize the following context: {context}",
                max_length=100,
                min_length=50,
                num_return_sequences=1
            )[0]["generated_text"]

        # Refine prompt
        prompt = (
            f"Question: {query}\n"
            f"Context: {context}\n"
            f"Explain the concept in clear and simple terms using one or two sentences."
        )

        # Generate answer
        answer = self.generator(prompt, max_length=80, min_length=10, num_return_sequences=1)
        refined_answer = self.refine_answer(answer[0]["generated_text"])

        return {"query": query, "answer": refined_answer, "retrieved_docs": retrieved_docs}


if __name__ == "__main__":
    # Paths for pretrained model
    pretrained_index = "faiss_index_pretrained.bin"
    pretrained_data = "prepared_data_pretrained.json"
    pretrained_model = "multi-qa-mpnet-base-dot-v1"

    # Paths for fine-tuned model
    finetuned_index = "faiss_index_finetuned.bin"
    finetuned_data = "prepared_data_finetuned.json"
    finetuned_model = "fine_tuned_model"

    # Get query from user
    query = input("Enter your query: ")

    # Pretrained model results
    print("\n--- Results from Pretrained Model ---")
    pretrained_rag = RAGModel(pretrained_index, pretrained_data, pretrained_model)
    pretrained_result = pretrained_rag.generate_answer(query)

    print(f"Query: {pretrained_result['query']}")
    print(f"Answer: {pretrained_result['answer']}")
    print("\nRetrieved Documents:")
    for doc in pretrained_result["retrieved_docs"]:
        print(f"- {doc['content']} (Distance: {doc['distance']:.2f})")

    # Fine-tuned model results
    print("\n--- Results from Fine-Tuned Model ---")
    finetuned_rag = RAGModel(finetuned_index, finetuned_data, finetuned_model)
    finetuned_result = finetuned_rag.generate_answer(query)

    print(f"Query: {finetuned_result['query']}")
    print(f"Answer: {finetuned_result['answer']}")
    print("\nRetrieved Documents:")
    for doc in finetuned_result["retrieved_docs"]:
        print(f"- {doc['content']} (Distance: {doc['distance']:.2f})")
