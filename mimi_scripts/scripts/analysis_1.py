import pandas as pd
import json
import os

# Create output directory
os.makedirs("analysis_results", exist_ok=True)

# Load the summarized dataframes
df_covid_train = pd.read_parquet("summarized_data/abstractive/covidqa_train_abstractive.parquet")
df_covid_test = pd.read_parquet("summarized_data/abstractive/covidqa_test_abstractive.parquet")
df_covid_validation = pd.read_parquet("summarized_data/abstractive/covidqa_validation_abstractive.parquet")

df_cuad_train = pd.read_parquet("summarized_data/abstractive/cuad_train_abstractive.parquet")
df_cuad_test = pd.read_parquet("summarized_data/abstractive/cuad_test_abstractive.parquet")
df_cuad_validation = pd.read_parquet("summarized_data/abstractive/cuad_validation_abstractive.parquet")

# Define datasets
datasets = {
    "COVIDQA": [df_covid_train, df_covid_test, df_covid_validation],
    "CUAD": [df_cuad_train, df_cuad_test, df_cuad_validation],
}

def analyze_dataframe(df, name):
    analysis = {}
    # Example question and average length of questions
    analysis["example_question"] = str(df["question"].iloc[0])  # Ensure it's a string
    analysis["average_question_length"] = float(df["question"].str.len().mean())
    # Example answer and average length of answers
    analysis["example_answer"] = str(df["response"].iloc[0])  # Ensure it's a string
    analysis["average_answer_length"] = float(df["response"].str.len().mean())
    # Example abstractive summaries
    analysis["example_summary"] = str(df["abstractive_summary"].iloc[0])  # Ensure it's a string
    analysis["average_summary_length"] = float(df["abstractive_summary"].str.len().mean())
    return analysis

# Analyze each dataframe
results = {}
for name, dfs in datasets.items():
    results[name] = {}
    for i, df in enumerate(dfs):
        set_name = ["Train", "Test", "Validation"][i]
        results[name][set_name] = analyze_dataframe(df, name)

# Save results to file
with open("analysis_results/abstractive_analysis.json", "w") as f:
    json.dump(results, f, indent=4)

print("Abstractive summarization analysis saved to abstractive_analysis.json")
