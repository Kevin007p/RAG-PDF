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

# Load summarized data
df_covid_train = pd.read_parquet("summarized_data/extractive_filtered/covidqa_train_extractive_filtered.parquet")
df_cuad_train = pd.read_parquet("summarized_data/extractive_filtered/cuad_train_extractive_filtered.parquet")
# df_fin_train = pd.read_parquet("summarized_data/extractive_filtered/finqa_train_extractive_filtered.parquet")

datasets = {
    "COVIDQA": df_covid_train,
    "CUAD": df_cuad_train,
    # "FINQA": df_fin_train
}

# Prepare and evaluate
results = {}
for name, df in datasets.items():
    print(f"Processing {name}...")
    extractive_data = prepare_t5_inputs(df, "extractive_summary")
    filtered_data = prepare_t5_inputs(df, "filtered_summary")
    
    for summary_type, data in [("Extractive", extractive_data), ("Filtered", filtered_data)]:
        bleu, rouge, time_taken, tokens = evaluate_and_measure(data, t5_model, t5_tokenizer, summary_type)
        results[f"{name}_{summary_type}"] = {
            "BLEU": bleu,
            "ROUGE": rouge,
            "Time (s)": time_taken,
            "Total Tokens": tokens
        }

# Save results
with open("generation_results/extractive_filtered_results.txt", "w") as f:
    for name, metrics in results.items():
        f.write(f"{name} T5 Evaluation ({summary_type}):\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value}\n")
        f.write("\n")

print("Extractive and Filtered evaluation results saved.")
