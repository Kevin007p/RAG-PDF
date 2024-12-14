import os
import time
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

# Define helper functions
def evaluate_retriever_accuracy(retriever, df, k_values, input_column):
    """
    Evaluates retrieval accuracy using Recall@k for different values of k.
    """
    correct_retrievals = {k: 0 for k in k_values}
    total_queries = len(df)

    for query, relevant_docs in zip(df['question'], df[input_column]):
        if not isinstance(relevant_docs, list):  # Skip invalid rows
            continue
        query_embedding = retriever.encode([query])
        documents = [" ".join(docs) for docs in df[input_column]]
        document_embeddings = retriever.encode(documents)

        # FAISS retrieval
        index = faiss.IndexFlatL2(document_embeddings.shape[1])
        index.add(np.array(document_embeddings, dtype="float32"))
        _, indices = index.search(np.array(query_embedding, dtype="float32"), max(k_values))

        for k in k_values:
            retrieved_docs = [documents[idx] for idx in indices[0][:k]]
            if any(doc in retrieved_docs for doc in relevant_docs):
                correct_retrievals[k] += 1

    recall_at_k = {k: correct_retrievals[k] / total_queries for k in k_values}
    return recall_at_k

def compute_average_document_size(retriever, df, input_column):
    """
    Computes the average size of documents (in tokens) queried.
    """
    tokenizer = retriever.tokenizer
    total_tokens = 0
    total_documents = 0

    for docs in df[input_column]:
        if isinstance(docs, list):
            for doc in docs:
                tokens = tokenizer.tokenize(doc) if isinstance(doc, str) else []
                total_tokens += len(tokens)
                total_documents += 1

    return total_tokens / total_documents if total_documents > 0 else 0

def evaluate_retriever(retriever, df, k_values, input_column):
    """
    Evaluates retriever performance including time, accuracy, and average document size.
    """
    start_time = time.time()
    recall_at_k = evaluate_retriever_accuracy(retriever, df, k_values, input_column)
    end_time = time.time()

    evaluation_time = end_time - start_time
    avg_doc_size = compute_average_document_size(retriever, df, input_column)

    return {
        "Recall@k": recall_at_k,
        "Time (s)": evaluation_time,
        "Average Document Size (tokens)": avg_doc_size
    }

# Load datasets
df_covidqa = pd.read_parquet("summarized_data/extractive_filtered/covidqa_test_extractive_filtered.parquet")
df_cuadqa = pd.read_parquet("summarized_data/extractive_filtered/cuad_test_extractive_filtered.parquet")

datasets = {
    "COVIDQA": df_covidqa,
    "CUADQA": df_cuadqa,
}

# Define retriever paths
retriever_paths = {
    "Extractive_COVID": "retriever_models/covidqa_extractive_retriever",
    "Filtered_COVID": "retriever_models/covidqa_filtered_retriever",
    "Extractive_CUAD": "retriever_models/cuad_extractive_retriever",
    "Filtered_CUAD": "retriever_models/cuad_filtered_retriever",
}

# Load retrievers
retrievers = {}
for name, path in retriever_paths.items():
    if not os.path.isdir(path):
        print(f"Error: Local model directory '{path}' does not exist. Please check the path.")
        continue
    print(f"Loading retriever: {name} from {path}")
    retrievers[name] = SentenceTransformer(path)

# Define parameters
k_values = [1, 4, 7]
results = {}

# Evaluate retrievers
for dataset_name, df in datasets.items():
    for retriever_name, retriever in retrievers.items():
        if dataset_name.split("_")[0] in retriever_name:  # Match dataset with appropriate retriever
            print(f"Evaluating {retriever_name} retriever on {dataset_name} dataset...")
            evaluation_result = evaluate_retriever(retriever, df, k_values, input_column="response")
            results[f"{dataset_name}_{retriever_name}"] = evaluation_result

# Save results to file
os.makedirs("retriever_evaluation_results", exist_ok=True)
output_file = "retriever_evaluation_results/retriever_performance_extractive_filtered.txt"

with open(output_file, "w") as f:
    for name, metrics in results.items():
        f.write(f"{name} Performance:\n")
        for metric, value in metrics.items():
            if metric == "Recall@k":
                f.write(f"{metric}:\n")
                for k, recall in value.items():
                    f.write(f"  Recall@{k}: {recall:.4f}\n")
            else:
                f.write(f"{metric}: {value:.2f}\n")
        f.write("\n")

print(f"Retriever evaluation results saved to {output_file}.")
