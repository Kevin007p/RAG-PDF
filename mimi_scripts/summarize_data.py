import pandas as pd

splits = {'train': 'covidqa/train-00000-of-00001.parquet', 'test': 'covidqa/test-00000-of-00001.parquet', 'validation': 'covidqa/validation-00000-of-00001.parquet'}
df_covid = pd.read_parquet("hf://datasets/rungalileo/ragbench/" + splits["train"])
df_covid_test = pd.read_parquet("hf://datasets/rungalileo/ragbench/" + splits["test"])
df_covid_validation = pd.read_parquet("hf://datasets/rungalileo/ragbench/" + splits["validation"])

splits = {'train': 'cuad/train-00000-of-00001.parquet', 'validation': 'cuad/validation-00000-of-00001.parquet', 'test': 'cuad/test-00000-of-00001.parquet'}
df_cuad = pd.read_parquet("hf://datasets/rungalileo/ragbench/" + splits["train"])
df_cuad_validation = pd.read_parquet("hf://datasets/rungalileo/ragbench/" + splits["validation"])
df_cuad_test = pd.read_parquet("hf://datasets/rungalileo/ragbench/" + splits["test"])

splits = {'train': 'finqa/train-00000-of-00001.parquet', 'validation': 'finqa/validation-00000-of-00001.parquet', 'test': 'finqa/test-00000-of-00001.parquet'}
df_fin = pd.read_parquet("hf://datasets/rungalileo/ragbench/" + splits["train"])
df_fin_test = pd.read_parquet("hf://datasets/rungalileo/ragbench/" + splits["test"])
df_fin_validation = pd.read_parquet("hf://datasets/rungalileo/ragbench/" + splits["validation"])

# Print and save the head and shape of each dataframe
with open('dataframes_info.txt', 'w') as f:
    for name, dfs in [("COVIDQA", [df_covid, df_covid_test, df_covid_validation]), 
                      ("CUAD", [df_cuad, df_cuad_test, df_cuad_validation]), 
                      ("FINQA", [df_fin, df_fin_test, df_fin_validation])]:
        for i, df in enumerate(dfs):
            set_name = ["Train", "Test", "Validation"][i]
            head = df.head()
            shape = df.shape
            
            print(f"{name} {set_name} Head:\n{head}\n")
            print(f"{name} {set_name} Shape: {shape}\n")
            
            f.write(f"{name} {set_name} Head:\n{head}\n\n")
            f.write(f"{name} {set_name} Shape: {shape}\n\n")

from transformers import pipeline

from summarizer import Summarizer

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download NLTK data if not already present
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

# Initialize Abstractive Summarizer
abstractive_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

extractive_summarizer = Summarizer()

# Generate abstractive summaries
def generate_abstractive_summaries(documents):
    """Generate abstractive summaries for a list of documents."""
    summaries = []
    for doc in documents:
        try:
            summary = abstractive_summarizer(doc, max_length=40, min_length=20, do_sample=False)[0]['summary_text']
            summaries.append(summary)
        except Exception as e:
            print(f"Error in abstractive summarization: {e}")
            summaries.append("")
    return summaries

# Generate extractive summaries
def generate_extractive_summaries(documents):
    return [extractive_summarizer(doc, ratio=0.6) for doc in documents]  # 60% of the text

# Generate filtered summaries (word removal)
def generate_filtered_summary(document):
    """Generate a filtered summary by removing stop words and non-alphabetic tokens."""
    try:
        # Tokenize and convert to lowercase
        tokens = word_tokenize(document.lower())
        
        # Define stop words
        stop_words = set(stopwords.words("english"))
        
        # Filter tokens
        filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
        
        # Join tokens back into a string
        return " ".join(filtered_tokens)
    except Exception as e:
        print(f"Error in filtered summarization: {e}")
        return ""

# Apply summarization functions to all datasets and all dataframes
datasets = {
    'COVIDQA': [df_covid, df_covid_test, df_covid_validation],
    'CUAD': [df_cuad, df_cuad_test, df_cuad_validation],
    'FINQA': [df_fin, df_fin_test, df_fin_validation]
}

for name, dfs in datasets.items():
    for i, df in enumerate(dfs):
        set_name = ["Train", "Test", "Validation"][i]
        print(f"Processing {name} {set_name} set...")

        df['raw_data'] = df['documents'].apply(lambda x: " ".join(x))
        df['abstractive_summary'] = df['documents'].apply(generate_abstractive_summaries)
        df['extractive_summary'] = df['documents'].apply(generate_extractive_summaries)
        df['filtered_summary'] = df['documents'].progress_apply(generate_filtered_summary)

        # Save the dataframe to a parquet file
        df.to_parquet(f"{name.lower()}_{set_name.lower()}_summarized.parquet")

import pandas as pd

# Load the summarized dataframes
df_covid_train_summarized = pd.read_parquet("covidqa_train_summarized.parquet")
df_covid_test_summarized = pd.read_parquet("covidqa_test_summarized.parquet")
df_covid_validation_summarized = pd.read_parquet("covidqa_validation_summarized.parquet")

df_cuad_train_summarized = pd.read_parquet("cuad_train_summarized.parquet")
df_cuad_test_summarized = pd.read_parquet("cuad_test_summarized.parquet")
df_cuad_validation_summarized = pd.read_parquet("cuad_validation_summarized.parquet")

df_fin_train_summarized = pd.read_parquet("finqa_train_summarized.parquet")
df_fin_test_summarized = pd.read_parquet("finqa_test_summarized.parquet")
df_fin_validation_summarized = pd.read_parquet("finqa_validation_summarized.parquet")

# Print samples to file
with open('summarized_samples.txt', 'w') as f:
    for name, dfs in [("COVIDQA", [df_covid_train_summarized, df_covid_test_summarized, df_covid_validation_summarized]), 
                      ("CUAD", [df_cuad_train_summarized, df_cuad_test_summarized, df_cuad_validation_summarized]), 
                      ("FINQA", [df_fin_train_summarized, df_fin_test_summarized, df_fin_validation_summarized])]:
        for i, df in enumerate(dfs):
            set_name = ["Train", "Test", "Validation"][i]
            sample = df.head(1)
            
            f.write(f"{name} {set_name} Sample:\n{sample}\n\n")
            f.write(f"{name} {set_name} Raw Data:\n{sample['raw_data'].values[0]}\n\n")

print("Samples saved to summarized_samples.txt")

import json

def analyze_dataframe(df, name):
    analysis = {}
    
    # Example question and average length of questions
    analysis['example_question'] = df['question'][0]
    analysis['average_question_length'] = df['question'].str.len().mean()
    
    # Example answer and average length of answers
    analysis['example_answer'] = df['response'][0]
    analysis['average_answer_length'] = df['response'].str.len().mean()
    
    # Example documents and average length of documents
    analysis['example_documents'] = df['documents'][0]
    document_lengths = [len(doc) for doc in df['documents'][0]]
    analysis['average_document_length'] = sum(document_lengths) / len(document_lengths)
    
    # Average number of documents per row
    analysis['average_number_of_documents'] = df['documents'].apply(len).mean()
    
    return analysis

# Analyze each dataframe set
datasets = {
    'COVIDQA': [df_covid, df_covid_test, df_covid_validation],
    'CUAD': [df_cuad, df_cuad_test, df_cuad_validation],
    'FINQA': [df_fin, df_fin_test, df_fin_validation]
}

results = {}
for name, dfs in datasets.items():
    results[name] = {}
    for i, df in enumerate(dfs):
        set_name = ["Train", "Test", "Validation"][i]
        results[name][set_name] = analyze_dataframe(df, name)

# Analyze summarized dataframes
summarized_datasets = {
    'COVIDQA': {
        'Train': df_covid_train_summarized,
        'Test': df_covid_test_summarized,
        'Validation': df_covid_validation_summarized
    },
    'CUAD': {
        'Train': df_cuad_train_summarized,
        'Test': df_cuad_test_summarized,
        'Validation': df_cuad_validation_summarized
    },
    'FINQA': {
        'Train': df_fin_train_summarized,
        'Test': df_fin_test_summarized,
        'Validation': df_fin_validation_summarized
    }
}

summarization_types = ['abstractive_summary', 'extractive_summary', 'filtered_summary']

for name, dfs in summarized_datasets.items():
    results[name + '_summarized'] = {}
    for set_name, df in dfs.items():
        results[name + '_summarized'][set_name] = {}
        for summary_type in summarization_types:
            results[name + '_summarized'][set_name][summary_type] = analyze_dataframe(df, name)

# Save the results to a file
with open('data_analysis_results.json', 'w') as f:
    json.dump(results, f, indent=4)

print("Analysis results saved to data_analysis_results.json")

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import numpy as np
import faiss

import pandas as pd

# Load the summarized dataframes
df_covid_train_summarized = pd.read_parquet("covidqa_train_summarized.parquet")
df_covid_test_summarized = pd.read_parquet("covidqa_test_summarized.parquet")
df_covid_validation_summarized = pd.read_parquet("covidqa_validation_summarized.parquet")

df_cuad_train_summarized = pd.read_parquet("cuad_train_summarized.parquet")
df_cuad_test_summarized = pd.read_parquet("cuad_test_summarized.parquet")
df_cuad_validation_summarized = pd.read_parquet("cuad_validation_summarized.parquet")

df_fin_train_summarized = pd.read_parquet("finqa_train_summarized.parquet")
df_fin_test_summarized = pd.read_parquet("finqa_test_summarized.parquet")
df_fin_validation_summarized = pd.read_parquet("finqa_validation_summarized.parquet")

# Combine all dataframes into a dictionary for easy access
summarized_datasets = {
    'COVIDQA': {
        'Train': df_covid_train_summarized,
        'Test': df_covid_test_summarized,
        'Validation': df_covid_validation_summarized
    },
    'CUAD': {
        'Train': df_cuad_train_summarized,
        'Test': df_cuad_test_summarized,
        'Validation': df_cuad_validation_summarized
    },
    'FINQA': {
        'Train': df_fin_train_summarized,
        'Test': df_fin_test_summarized,
        'Validation': df_fin_validation_summarized
    }
}

print("All summarized datasets loaded successfully.")

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

# Generate datasets for each setup
raw_data_covidqa = prepare_retrieval_training_data(summarized_datasets['COVIDQA']['Train'], "documents", "response")
abstractive_data_covidqa = prepare_retrieval_training_data(summarized_datasets['COVIDQA']['Train'], "abstractive_summary", "response")
extractive_data_covidqa = prepare_retrieval_training_data(summarized_datasets['COVIDQA']['Train'], "extractive_summary", "response")
filtered_data_covidqa = prepare_retrieval_training_data(summarized_datasets['COVIDQA']['Train'], "filtered_summary", "response")

raw_data_cuad = prepare_retrieval_training_data(summarized_datasets['CUAD']['Train'], "documents", "response")
abstractive_data_cuad = prepare_retrieval_training_data(summarized_datasets['CUAD']['Train'], "abstractive_summary", "response")
extractive_data_cuad = prepare_retrieval_training_data(summarized_datasets['CUAD']['Train'], "extractive_summary", "response")
filtered_data_cuad = prepare_retrieval_training_data(summarized_datasets['CUAD']['Train'], "filtered_summary", "response")

raw_data_finqa = prepare_retrieval_training_data(summarized_datasets['FINQA']['Train'], "documents", "response")
abstractive_data_finqa = prepare_retrieval_training_data(summarized_datasets['FINQA']['Train'], "abstractive_summary", "response")
extractive_data_finqa = prepare_retrieval_training_data(summarized_datasets['FINQA']['Train'], "extractive_summary", "response")
filtered_data_finqa = prepare_retrieval_training_data(summarized_datasets['FINQA']['Train'], "filtered_summary", "response")

# Fine-tune the model for retrieval
def fine_tune_retriever(train_data, model_name, output_path):
    """
    Fine-tune the retriever using MultipleNegativesRankingLoss.
    """
    model = SentenceTransformer(model_name)
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=16)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # Train
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=3,
        warmup_steps=100,
        evaluator=None,  # Add an evaluator if needed
        evaluation_steps=1000,  # Evaluate every 1000 steps
        output_path=output_path
    )
    model.save(output_path)
    return model

# Fine-tune for each dataset and summarization type
raw_retriever_covidqa = fine_tune_retriever(raw_data_covidqa, "all-MiniLM-L6-v2", "fine_tuned_retriever_raw_covidqa")
abstractive_retriever_covidqa = fine_tune_retriever(abstractive_data_covidqa, "all-MiniLM-L6-v2", "fine_tuned_retriever_abstractive_covidqa")
extractive_retriever_covidqa = fine_tune_retriever(extractive_data_covidqa, "all-MiniLM-L6-v2", "fine_tuned_retriever_extractive_covidqa")
filtered_retriever_covidqa = fine_tune_retriever(filtered_data_covidqa, "all-MiniLM-L6-v2", "fine_tuned_retriever_filtered_covidqa")

raw_retriever_cuad = fine_tune_retriever(raw_data_cuad, "all-MiniLM-L6-v2", "fine_tuned_retriever_raw_cuad")
abstractive_retriever_cuad = fine_tune_retriever(abstractive_data_cuad, "all-MiniLM-L6-v2", "fine_tuned_retriever_abstractive_cuad")
extractive_retriever_cuad = fine_tune_retriever(extractive_data_cuad, "all-MiniLM-L6-v2", "fine_tuned_retriever_extractive_cuad")
filtered_retriever_cuad = fine_tune_retriever(filtered_data_cuad, "all-MiniLM-L6-v2", "fine_tuned_retriever_filtered_cuad")

raw_retriever_finqa = fine_tune_retriever(raw_data_finqa, "all-MiniLM-L6-v2", "fine_tuned_retriever_raw_finqa")
abstractive_retriever_finqa = fine_tune_retriever(abstractive_data_finqa, "all-MiniLM-L6-v2", "fine_tuned_retriever_abstractive_finqa")
extractive_retriever_finqa = fine_tune_retriever(extractive_data_finqa, "all-MiniLM-L6-v2", "fine_tuned_retriever_extractive_finqa")
filtered_retriever_finqa = fine_tune_retriever(filtered_data_finqa, "all-MiniLM-L6-v2", "fine_tuned_retriever_filtered_finqa")


def evaluate_retriever_accuracy(retriever, df, k_values, input_column):
    """
    Evaluates retrieval accuracy using Recall@k for different values of k.
    """
    correct_retrievals = {k: 0 for k in k_values}
    total_queries = len(df)

    for query, relevant_docs in zip(df['question'], df[input_column]):
        query_embedding = retriever.encode([query])
        documents = [" ".join(docs) for docs in df[input_column]]
        document_embeddings = retriever.encode(documents)

        distances = faiss.IndexFlatL2(document_embeddings.shape[1])
        distances.add(np.array(document_embeddings, dtype="float32"))
        _, indices = distances.search(np.array(query_embedding, dtype="float32"), max(k_values))

        for k in k_values:
            retrieved_docs = [documents[idx] for idx in indices[0][:k]]
            if any(doc in retrieved_docs for doc in relevant_docs):
                correct_retrievals[k] += 1

    recall_at_k = {k: correct_retrievals[k] / total_queries for k in k_values}
    return recall_at_k

# Evaluate each retriever and print results to file
k_values = [1, 4, 7]
retrievers = {
    "Raw": raw_retriever_covidqa,
    "Abstractive": abstractive_retriever_covidqa,
    "Extractive": extractive_retriever_covidqa,
    "Filtered": filtered_retriever_covidqa
}

datasets = {
    "COVIDQA": summarized_datasets['COVIDQA']['Train'],
    "CUAD": summarized_datasets['CUAD']['Train'],
    "FINQA": summarized_datasets['FINQA']['Train']
}

with open('retriever_accuracy_results.txt', 'w') as f:
    for retriever_name, retriever in retrievers.items():
        for dataset_name, df in datasets.items():
            for summary_type in ["documents", "abstractive_summary", "extractive_summary", "filtered_summary"]:
                accuracy = evaluate_retriever_accuracy(retriever, df, k_values, summary_type)
                f.write(f"{retriever_name} Retriever on {dataset_name} ({summary_type}):\n")
                f.write(f"Recall@k: {accuracy}\n\n")

print("Retriever accuracy results saved to retriever_accuracy_results.txt")


from transformers import T5Tokenizer, T5ForConditionalGeneration
import time

# Load the model and tokenizer
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Prepare evaluation data
def prepare_t5_inputs(df, input_column):
    return [
        {"input_text": f"question: {query} context: {context}", "reference": answer}
        for query, context, answer in zip(df['question'], df[input_column], df['response'])
    ]

def evaluate_t5_model(data, t5_model, t5_tokenizer):
    """
    Evaluates the T5 model's answer quality using BLEU and ROUGE scores.
    """
    bleu_metric = load("bleu")
    rouge_metric = load("rouge")
    predictions = []
    references = []

    for sample in data:
        inputs = t5_tokenizer(
            sample["input_text"],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512
        )
        outputs = t5_model.generate(inputs['input_ids'], max_length=128, min_length=10)
        predictions.append(t5_tokenizer.decode(outputs[0], skip_special_tokens=True))
        references.append(sample["reference"])

    bleu_score = bleu_metric.compute(predictions=predictions, references=[[ref] for ref in references])
    rouge_score = rouge_metric.compute(predictions=predictions, references=references)

    return bleu_score, rouge_score


# Function to evaluate and measure time and token usage
def evaluate_and_measure(data, t5_model, t5_tokenizer, description):
    start_time = time.time()
    bleu_score, rouge_score = evaluate_t5_model(data, t5_model, t5_tokenizer)
    end_time = time.time()
    total_time = end_time - start_time
    total_tokens = sum(len(t5_tokenizer(sample["input_text"])['input_ids']) for sample in data)
    return bleu_score, rouge_score, total_time, total_tokens

# Prepare data and evaluate each setup
raw_t5_data = prepare_t5_inputs(df, "documents")
abstractive_t5_data = prepare_t5_inputs(df, "abstractive_summary")
extractive_t5_data = prepare_t5_inputs(df, "extractive_summary")
filtered_t5_data = prepare_t5_inputs(df, "filtered_summary")

# Evaluate and measure for each setup
results = {}
for name, data in [("Raw", raw_t5_data), ("Abstractive", abstractive_t5_data), ("Extractive", extractive_t5_data), ("Filtered", filtered_t5_data)]:
    print(f"Evaluating {name} data...")
    bleu, rouge, time_taken, tokens = evaluate_and_measure(data, t5_model, t5_tokenizer, name)
    results[name] = {
        "BLEU": bleu,
        "ROUGE": rouge,
        "Time (s)": time_taken,
        "Total Tokens": tokens
    }

# Print results to file
with open('t5_evaluation_results.txt', 'w') as f:
    for name, metrics in results.items():
        f.write(f"{name} T5 Evaluation:\n")
        f.write(f"BLEU: {metrics['BLEU']}\n")
        f.write(f"ROUGE: {metrics['ROUGE']}\n")
        f.write(f"Time (s): {metrics['Time (s)']}\n")
        f.write(f"Total Tokens: {metrics['Total Tokens']}\n\n")

print("T5 evaluation results saved to t5_evaluation_results.txt")



