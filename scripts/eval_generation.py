from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
import time
import os
from evaluate import load

# Load the model and tokenizer
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Create output directory
os.makedirs("generation_results", exist_ok=True)

# Prepare evaluation data
def prepare_t5_inputs(df, input_column):
    return [
        {"input_text": f"question: {query} context: {context}", "reference": answer}
        for query, context, answer in zip(df['question'], df[input_column], df['response'])
    ]

# Function to evaluate the T5 model
def evaluate_t5_model(data, t5_model, t5_tokenizer):
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

# Load datasets
dataset_paths = {
    "COVIDQA_Extractive": "summarized_data/extractive_filtered/covidqa_test_extractive_filtered.parquet",
    "CUAD_Extractive": "summarized_data/extractive_filtered/cuad_test_extractive_filtered.parquet",
    "COVIDQA_Abstractive": "summarized_data/abstractive/covidqa_test_abstractive.parquet",
    "CUAD_Abstractive": "summarized_data/abstractive/cuad_test_abstractive.parquet",
    # "FINQA_Extractive": "summarized_data/extractive_filtered/finqa_train_extractive_filtered.parquet",
    # "FINQA_Abstractive": "summarized_data/abstractive/finqa_train_abstractive.parquet",
}

datasets = {}
for name, path in dataset_paths.items():
    try:
        datasets[name] = pd.read_parquet(path)
    except FileNotFoundError as e:
        print(f"Error loading dataset {name}: {e}")

# Prepare and evaluate
results = {}
for name, df in datasets.items():
    print(f"Processing {name}...")
    
    # Dynamically determine columns based on dataset type
    if "Extractive" in name:
        input_columns = [("Extractive", "extractive_summary"), ("Filtered", "filtered_summary")]
    elif "Abstractive" in name:
        input_columns = [("Raw", "documents"), ("Abstractive", "abstractive_summary")]
    else:
        input_columns = []

    for summary_type, column in input_columns:
        data = prepare_t5_inputs(df, column)
        bleu, rouge, time_taken, tokens = evaluate_and_measure(data, t5_model, t5_tokenizer, summary_type)
        results[f"{name}_{summary_type}"] = {
            "BLEU": bleu,
            "ROUGE": rouge,
            "Time (s)": time_taken,
            "Total Tokens": tokens
        }

# Save results
output_file = "generation_results/t5_evaluation_results.txt"
with open(output_file, "w") as f:
    for name, metrics in results.items():
        f.write(f"{name} T5 Evaluation:\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value}\n")
        f.write("\n")

print(f"Evaluation results saved to {output_file}.")
