import os
import pandas as pd
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses

# Create output directory
os.makedirs("retriever_models", exist_ok=True)

# Load datasets
df_covid_train = pd.read_parquet("summarized_data/abstractive/covidqa_train_abstractive.parquet")
df_cuad_train = pd.read_parquet("summarized_data/abstractive/cuad_train_abstractive.parquet")

# Define datasets
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
    "COVID_Raw": prepare_retrieval_training_data(df_covid_train, "documents", "response"),
    "COVID_Abstractive": prepare_retrieval_training_data(df_covid_train, "abstractive_summary", "response"),
    "CUAD_Raw": prepare_retrieval_training_data(df_cuad_train, "documents", "response"),
    "CUAD_Abstractive": prepare_retrieval_training_data(df_cuad_train, "abstractive_summary", "response"),
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
    try:
        model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=3, warmup_steps=100)
        model.save(output_path)
        print(f"Retriever saved to {output_path}")
    except Exception as e:
        print(f"Error during training {output_path}: {e}")

# Train retrievers
for name, train_data in training_data.items():
    print(f"Processing {name} retriever...")
    if len(train_data) == 0:
        print(f"Skipping {name} due to lack of data.")
        continue
    model_path = f"retriever_models/{name.lower()}_retriever"
    fine_tune_retriever(train_data, "all-MiniLM-L6-v2", model_path)

print("All retrievers trained and saved.")
