import pandas as pd
import json
import os

# Create output directory
os.makedirs("analysis_results", exist_ok=True)

# Load the extractive and filtered summarized dataframes
df_covid_train = pd.read_parquet("summarized_data/extractive_filtered/covidqa_train_extractive_filtered.parquet")
df_covid_test = pd.read_parquet("summarized_data/extractive_filtered/covidqa_test_extractive_filtered.parquet")
df_covid_validation = pd.read_parquet("summarized_data/extractive_filtered/covidqa_validation_extractive_filtered.parquet")

df_cuad_train = pd.read_parquet("summarized_data/extractive_filtered/cuad_train_extractive_filtered.parquet")
df_cuad_test = pd.read_parquet("summarized_data/extractive_filtered/cuad_test_extractive_filtered.parquet")
df_cuad_validation = pd.read_parquet("summarized_data/extractive_filtered/cuad_validation_extractive_filtered.parquet")

# Define datasets
datasets = {
    "COVIDQA": [df_covid_train, df_covid_test, df_covid_validation],
    "CUAD": [df_cuad_train, df_cuad_test, df_cuad_validation],
}

def analyze_dataframe(df, name):
    analysis = {}
    # Example question and average length of questions
    analysis["example_question"] = str(df["question"].iloc[0])  # Ensure string
    analysis["average_question_length"] = float(df["question"].str.len().mean())  # Ensure float
    # Example answer and average length of answers
    analysis["example_answer"] = str(df["response"].iloc[0])  # Ensure string
    analysis["average_answer_length"] = float(df["response"].str.len().mean())  # Ensure float
    # Example extractive summaries
    analysis["example_extractive_summary"] = str(df["extractive_summary"].iloc[0])  # Ensure string
    analysis["average_extractive_summary_length"] = float(df["extractive_summary"].str.len().mean())  # Ensure float
    # Example filtered summaries
    analysis["example_filtered_summary"] = str(df["filtered_summary"].iloc[0])  # Ensure string
    analysis["average_filtered_summary_length"] = float(df["filtered_summary"].str.len().mean())  # Ensure float
    return analysis

# Analyze each dataframe
results = {}
for name, dfs in datasets.items():
    results[name] = {}
    for i, df in enumerate(dfs):
        set_name = ["Train", "Test", "Validation"][i]
        results[name][set_name] = analyze_dataframe(df, name)

# Save results to file
with open("analysis_results/extractive_filtered_analysis.json", "w") as f:
    json.dump(results, f, indent=4)

print("Extractive and filtered summarization analysis saved to extractive_filtered_analysis.json")
