import pandas as pd
from transformers import pipeline
import os
from summarizer import Summarizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download NLTK data
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

# Create output directory
os.makedirs("summarized_data", exist_ok=True)

# Define dataset splits
datasets = {
    "COVIDQA": {
        "train": "hf://datasets/rungalileo/ragbench/covidqa/train-00000-of-00001.parquet",
        "test": "hf://datasets/rungalileo/ragbench/covidqa/test-00000-of-00001.parquet",
        "validation": "hf://datasets/rungalileo/ragbench/covidqa/validation-00000-of-00001.parquet",
    }
}

# Load the abstractive summarization model
abstractive_summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    tokenizer="facebook/bart-large-cnn",
    device=0,
)

# Initialize extractive summarizer and stopwords
extractive_summarizer = Summarizer()
stop_words = set(stopwords.words("english"))


# Truncate document if too long for the abstractive summarizer
def truncate_document(document, max_tokens=1024):
    tokenizer = abstractive_summarizer.tokenizer
    tokens = tokenizer(
        document, truncation=True, max_length=max_tokens, return_tensors="pt"
    )
    return tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)


# Abstractive Summarization for each document in a list
def generate_abstractive_summaries(documents):
    summaries = []
    for doc in documents:
        if isinstance(doc, str) and len(doc.strip()) > 0:
            try:
                truncated_doc = truncate_document(doc)  # Truncate document if necessary
                summary = abstractive_summarizer(
                    truncated_doc, max_length=50, min_length=20, do_sample=False
                )[0]["summary_text"]
                summaries.append(summary)
            except Exception as e:
                print(f"Error in abstractive summarization: {e}")
                summaries.append("")
        else:
            summaries.append("")
    return summaries


# Extractive Summarization for each document in a list
def generate_extractive_summaries(documents):
    return [
        extractive_summarizer(doc, ratio=0.3) if isinstance(doc, str) else ""
        for doc in documents
    ]


# Filtered summaries by removing stopwords
def generate_filtered_summaries(documents):
    filtered_summaries = []
    for doc in documents:
        if isinstance(doc, str):
            tokens = word_tokenize(doc.lower())
            filtered_tokens = [
                word for word in tokens if word.isalpha() and word not in stop_words
            ]
            filtered_summaries.append(" ".join(filtered_tokens))
        else:
            filtered_summaries.append("")
    return filtered_summaries


# Process datasets
for dataset_name, splits in datasets.items():
    for split_name, file_path in splits.items():
        print(f"Processing {dataset_name} {split_name}...")

        try:
            # Load the dataset
            df = pd.read_parquet(file_path)

            # Ensure 'raw_data' contains the original 3 documents as a list
            df["raw_data"] = df["documents"].apply(
                lambda docs: docs if isinstance(docs, list) else [str(docs)]
            )

            # Generate summaries
            df["abstractive_summary"] = df["documents"].apply(
                generate_abstractive_summaries
            )
            df["extractive_summary"] = df["documents"].apply(
                generate_extractive_summaries
            )
            df["filtered_summary"] = df["documents"].apply(generate_filtered_summaries)

            # Retain necessary columns
            df = df[
                [
                    "question",
                    "response",
                    "raw_data",
                    "abstractive_summary",
                    "extractive_summary",
                    "filtered_summary",
                    "relevance_score",
                ]
            ]

            # Save the summarized data
            output_path = f"summarized_data/{dataset_name}_{split_name}.parquet"
            df.to_parquet(output_path)
            print(f"Saved summarized data to {output_path}")
        except Exception as e:
            print(f"Error processing {dataset_name} {split_name}: {e}")
