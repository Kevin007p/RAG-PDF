import pandas as pd
from transformers import pipeline
import os

# Create output directory
os.makedirs("summarized_data/abstractive", exist_ok=True)

# Define dataset splits
datasets = {
    'COVIDQA': {
        'train': 'hf://datasets/rungalileo/ragbench/covidqa/train-00000-of-00001.parquet',
        'test': 'hf://datasets/rungalileo/ragbench/covidqa/test-00000-of-00001.parquet',
        'validation': 'hf://datasets/rungalileo/ragbench/covidqa/validation-00000-of-00001.parquet'
    },
    'CUAD': {
        'train': 'hf://datasets/rungalileo/ragbench/cuad/train-00000-of-00001.parquet',
        'test': 'hf://datasets/rungalileo/ragbench/cuad/test-00000-of-00001.parquet',
        'validation': 'hf://datasets/rungalileo/ragbench/cuad/validation-00000-of-00001.parquet'
    },
    # 'FINQA': {
    #     'train': 'hf://datasets/rungalileo/ragbench/finqa/train-00000-of-00001.parquet',
    #     'test': 'hf://datasets/rungalileo/ragbench/finqa/test-00000-of-00001.parquet',
    #     'validation': 'hf://datasets/rungalileo/ragbench/finqa/validation-00000-of-00001.parquet'
    # }
}

# Load the abstractive summarization model
abstractive_summarizer = pipeline("summarization", model="facebook/bart-large-cnn", tokenizer="facebook/bart-large-cnn", device=0)

# Function to truncate long documents using the tokenizer
def truncate_document(document, max_tokens=1024):
    tokenizer = abstractive_summarizer.tokenizer
    tokens = tokenizer(document, truncation=True, max_length=max_tokens, return_tensors="pt")
    truncated_text = tokenizer.decode(tokens['input_ids'][0], skip_special_tokens=True)
    return truncated_text

# Generate summaries with truncation
def generate_abstractive_summaries(documents):
    summaries = []
    for doc in documents:
        if not doc or len(doc.strip()) == 0:  # Skip empty documents
            summaries.append("")
            continue
        try:
            # Truncate document if necessary
            truncated_doc = truncate_document(doc)
            # Generate summary
            summary = abstractive_summarizer(
                truncated_doc, max_length=50, min_length=20, do_sample=False
            )[0]['summary_text']
            summaries.append(summary)
        except Exception as e:
            print(f"Error in abstractive summarization: {e}")
            summaries.append("")
    return summaries

# Process and summarize datasets
for dataset_name, splits in datasets.items():
    for split_name, file_path in splits.items():
        print(f"Processing {dataset_name} {split_name}...")
        try:
            # Load the dataset
            df = pd.read_parquet(file_path)
            
            # Prepare raw data by joining documents into single strings
            df['raw_data'] = df['documents'].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))
            
            # Generate abstractive summaries
            df['abstractive_summary'] = df['raw_data'].apply(lambda x: generate_abstractive_summaries([x])[0])
            
            # Save the summarized data
            output_path = f"summarized_data/abstractive/{dataset_name.lower()}_{split_name.lower()}_abstractive.parquet"
            df.to_parquet(output_path)
            print(f"Saved summarized data to {output_path}")
        except Exception as e:
            print(f"Error processing {dataset_name} {split_name}: {e}")
