import pandas as pd

dataset_name = "COVIDQA"
# dataset_name = "emanual"

# Paths for the dataset splits
splits = {
    "train": f"summarized_data/{dataset_name}_train.parquet",
    "validation": f"summarized_data/{dataset_name}_validation.parquet",
    "test": f"summarized_data/{dataset_name}_test.parquet",
}

# List of summarization types to process
summarization_types = [
    "raw_data",
    "abstractive_summary",
    "extractive_summary",
    "filtered_summary",
]


for summarization_type in summarization_types:

    # Initialize lists for the consolidated document database
    all_doc_ids = []
    all_doc_contents = []

    # Process each split
    for split_name, split_path in splits.items():
        print(f"Processing split: {split_name}")

        # Load the split
        split_data = pd.read_parquet(split_path)

        # id in dataset is train-01-13
        # id column name in dataset is raw_data_doc_ids
        split_data[f"{summarization_type}_doc_ids"] = [
            [f"{split_name}-{idx}-{doc_idx}" for doc_idx in range(len(docs))]
            for idx, docs in enumerate(split_data[summarization_type])
        ]

        print(f"Added '{summarization_type}_doc_ids' to {split_name} split.")

        # Save the updated split
        split_data.to_parquet(split_path, index=False)
        print(f"Updated {split_name} split saved to {split_path}")

        # id in database is train-01-13
        # id column name in database is doc_id
        for idx, docs in enumerate(split_data[summarization_type]):
            for doc_idx, content in enumerate(docs):
                all_doc_ids.append(f"{split_name}-{idx}-{doc_idx}")
                all_doc_contents.append(content)

    # Create the global document database
    doc_database = pd.DataFrame({"doc_id": all_doc_ids, "content": all_doc_contents})

    # Remove duplicates (as different summarization types can have the same IDs with different content)
    doc_database = doc_database.drop_duplicates(
        subset=["doc_id", "content"]
    ).reset_index(drop=True)

    # Save the document database
    doc_database_path = (
        f"summarized_data/{dataset_name}_{summarization_type}_database.parquet"
    )
    doc_database.to_parquet(doc_database_path, index=False)
    print(f"Document database saved to {doc_database_path}")
