import os
import pandas as pd
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses

# Create output directory
os.makedirs("retriever_models", exist_ok=True)

# Load datasets
try:
    df_covid_train = pd.read_parquet("summarized_data/extractive_filtered/covidqa_train_extractive_filtered.parquet")
    df_cuad_train = pd.read_parquet("summarized_data/extractive_filtered/cuad_train_extractive_filtered.parquet")
except FileNotFoundError as e:
    print(f"File not found: {e}")
    exit(1)

datasets = {
    "COVIDQA": df_covid_train,
    "CUAD": df_cuad_train,
}

# Prepare data for fine-tuning
def prepare_retrieval_training_data(df, input_column, target_column):
    """
    Prepares training data for retrieval fine-tuning.
    Each query is paired with its corresponding document (positive example).
    """
    positive_pairs = [
        InputExample(texts=[query, " ".join(docs)], label=1.0) 
        for query, docs in zip(df['question'], df[input_column])
    ]
    return positive_pairs

# Prepare training data
training_data = {
    "COVID_Extractive": prepare_retrieval_training_data(df_covid_train, "extractive_summary", "response"),
    "CUAD_Extractive": prepare_retrieval_training_data(df_cuad_train, "extractive_summary", "response"),
    "COVID_Filtered": prepare_retrieval_training_data(df_covid_train, "filtered_summary", "response"),
    "CUAD_Filtered": prepare_retrieval_training_data(df_cuad_train, "filtered_summary", "response"),
}

def fine_tune_retriever(train_data, model_name, output_path):
    """
    Fine-tunes a retriever model with the provided training data.
    """
    if not train_data:
        print(f"No training data provided for {output_path}. Skipping fine-tuning.")
        return

    model = SentenceTransformer(model_name)
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=16)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    print(f"Training retriever for {output_path}...")
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=3, warmup_steps=100)
    model.save(output_path)
    print(f"Retriever saved to {output_path}")

# Train retrievers
for summary_type, train_data in training_data.items():
    if not train_data:
        print(f"Skipping {summary_type} retriever: No training data.")
        continue
    model_path = f"retriever_models/{summary_type.lower()}_retriever"
    fine_tune_retriever(train_data, "all-MiniLM-L6-v2", model_path)

print("Extractive and Filtered retrievers trained and saved.")
