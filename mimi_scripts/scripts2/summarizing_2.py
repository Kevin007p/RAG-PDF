import pandas as pd
from summarizer import Summarizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
import nltk

# Download NLTK data
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

# Create output directory
os.makedirs("summarized_data/extractive_filtered", exist_ok=True)

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
    'FINQA': {
        'train': 'hf://datasets/rungalileo/ragbench/finqa/train-00000-of-00001.parquet',
        'test': 'hf://datasets/rungalileo/ragbench/finqa/test-00000-of-00001.parquet',
        'validation': 'hf://datasets/rungalileo/ragbench/finqa/validation-00000-of-00001.parquet'
    }
}

# Initialize extractive summarizer and stopwords
extractive_summarizer = Summarizer()
stop_words = set(stopwords.words("english"))

def generate_extractive_summaries(documents):
    return [extractive_summarizer(doc, ratio=0.3) for doc in documents]

def generate_filtered_summary(document):
    try:
        tokens = word_tokenize(document.lower())
        filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
        return " ".join(filtered_tokens)
    except Exception as e:
        print(f"Error in filtered summarization: {e}")
        return ""

# Process and summarize datasets
for dataset_name, splits in datasets.items():
    for split_name, file_path in splits.items():
        print(f"Processing {dataset_name} {split_name}...")
        df = pd.read_parquet(file_path)
        df['extractive_summary'] = df['documents'].apply(generate_extractive_summaries)
        df['filtered_summary'] = df['documents'].apply(lambda x: generate_filtered_summary(" ".join(x)))
        df.to_parquet(f"summarized_data/extractive_filtered/{dataset_name.lower()}_{split_name.lower()}_extractive_filtered.parquet")
