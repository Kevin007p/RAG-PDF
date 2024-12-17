import time
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
import pandas as pd
import json


class EmbeddingRetrievalEvaluator:
    def __init__(self, model):
        """
        Initializes the embedding retriever evaluator with a pretrained model.

        Args:
            model (SentenceTransformer): Pretrained SentenceTransformer model.
        """
        self.model = model
        self.index = None
        self.dataset = None

    def embed_dataset(self, dataset, text_column, id_column="doc_id", batch_size=128):
        """
        Embeds the dataset in batches and stores it in a FAISS index.

        Args:
            dataset (pd.DataFrame): The dataset containing documents to embed.
            text_column (str): Column name in the dataset containing the text to embed.
            id_column (str): Column name in the dataset containing the document IDs.
            batch_size (int): Number of documents to process per batch.
        """
        self.dataset = dataset
        self.text_column = text_column
        self.id_column = id_column

        # Initialize FAISS index
        print("Initializing FAISS index...")
        embedding_dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(embedding_dimension)

        # Encode documents in batches
        print("Embedding documents in batches...")
        documents = dataset[text_column].tolist()
        for start_idx in range(0, len(documents), batch_size):
            batch = documents[start_idx : start_idx + batch_size]
            embeddings = self.model.encode(batch, convert_to_numpy=True)
            self.index.add(embeddings)

        print(f"Embedded {len(documents)} documents.")

    def evaluate_retrieval(self, test_set, query_column, relevant_column, k=5):
        """
        Evaluates the retrieval performance on the test set.

        Args:
            test_set (pd.DataFrame): Test set containing queries and ground truth document IDs.
            query_column (str): Column name for the query text.
            relevant_column (str): Column name for the relevant ground truth document IDs.
            k (int): Number of documents to retrieve for each query.

        Returns:
            dict: Retrieval metrics including precision, recall, and retrieval time.
        """
        if self.index is None or self.dataset is None:
            raise ValueError("FAISS index is not initialized. Run embed_dataset first.")

        # Encode test queries
        print("Evaluating retrieval...")
        test_queries = test_set[query_column].tolist()
        query_embeddings = self.model.encode(test_queries, convert_to_numpy=True)

        # Evaluate retrieval
        precision_list = []
        recall_list = []
        retrieval_times = []

        for i, query_embedding in enumerate(query_embeddings):
            # Relevant doc_ids as a set
            relevant_docs = set(test_set.iloc[i][relevant_column])
            num_relevant_docs = len(relevant_docs)

            if num_relevant_docs == 0:
                continue

            # Measure retrieval time
            start_time = time.time()
            distances, indices = self.index.search(np.array([query_embedding]), k)
            retrieval_time = time.time() - start_time
            retrieval_times.append(retrieval_time)

            # Retrieved document IDs
            retrieved_ids = set(self.dataset.iloc[indices[0]][self.id_column].tolist())

            # Precision and Recall
            true_positives = len(relevant_docs & retrieved_ids)
            precision = true_positives / len(retrieved_ids) if retrieved_ids else 0
            recall = true_positives / num_relevant_docs

            precision_list.append(precision)
            recall_list.append(recall)

        # Aggregate results
        metrics = {
            "Precision@k": np.mean(precision_list) if precision_list else 0,
            "Recall@k": np.mean(recall_list) if recall_list else 0,
            "Avg Retrieval Time (s)": (
                np.mean(retrieval_times) if retrieval_times else 0
            ),
        }
        return metrics


class EmbeddingTrainer:
    def __init__(self, model):
        """
        Initializes the embedding trainer with a pretrained model.

        Args:
            model (SentenceTransformer): Pretrained SentenceTransformer model.
        """
        self.model = model

    def load_data(self, dataset, x_col, y_col):
        """
        Loads the dataset and prepares it for training.

        Args:
            dataset (pd.DataFrame): The dataset containing query and document columns.
            x_col (str): Column name for the query or input text.
            y_col (str): Column name for the document or summary text.
        """
        self.dataset = dataset
        self.x_col = x_col
        self.y_col = y_col

        # Create training examples
        self.train_examples = [
            InputExample(texts=[row[x_col], doc])
            for _, row in dataset.iterrows()
            for doc in row[y_col]  # Iterate through each document in the list
        ]
        print(f"Loaded {len(self.train_examples)} training examples.")

    def fine_tune(self, batch_size=16, epochs=10):
        """
        Fine-tunes the embedding model on the provided dataset.

        Args:
            batch_size (int): Batch size for training.
            epochs (int): Number of training epochs.

        Returns:
            SentenceTransformer: The fine-tuned model.
        """
        if not hasattr(self, "train_examples"):
            raise ValueError(
                "No data loaded. Use load_data method to load the dataset first."
            )

        # Create DataLoader
        train_dataloader = DataLoader(
            self.train_examples, shuffle=True, batch_size=batch_size
        )

        # Define the loss function
        train_loss = losses.MultipleNegativesRankingLoss(self.model)

        # Fine-tune the model
        print("Starting fine-tuning...")
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
        )
        print("Fine-tuning completed.")
        return self.model


def train_and_evaluate(
    dataset_path_train,
    dataset_path_test,
    dataset,
    summarization_types,
    k_values,
    output_metrics_path,
):
    """
    Trains and evaluates a model for each summarization type in the dataset.

    Args:
        dataset_path_train (str): Path to the training dataset file.
        dataset_path_test (str): Path to the testing dataset file.
        summarization_types (list): List of summarization types to train and evaluate on.
        k_values (list): List of k values for evaluation.
        output_metrics_path (str): Path to save the evaluation metrics as a JSON file.
    """
    # Load datasets
    train_df = pd.read_parquet(dataset_path_train)
    test_df = pd.read_parquet(dataset_path_test)

    # Dictionary to store metrics for each summarization type and k value
    all_metrics = {}

    for summarization_type in summarization_types:
        print(
            f"\nTraining and evaluating for summarization type: {summarization_type}\n"
        )
        all_metrics[summarization_type] = {}

        # Initialize and fine-tune the model
        base_model = SentenceTransformer("all-MiniLM-L6-v2")
        trainer = EmbeddingTrainer(model=base_model)
        trainer.load_data(train_df, x_col="question", y_col=summarization_type)
        fine_tuned_model = trainer.fine_tune()

        # Initialize the evaluator with the fine-tuned model
        evaluator = EmbeddingRetrievalEvaluator(model=fine_tuned_model)

        database_path = (
            f"summarized_data/{dataset}_{summarization_type}_database.parquet"
        )
        database_df = pd.read_parquet(database_path)

        # Embed the document database
        evaluator.embed_dataset(
            dataset=database_df, text_column="content", id_column="doc_id"
        )

        # Evaluate on the test set for different k values
        for k in k_values:
            metrics = evaluator.evaluate_retrieval(
                test_set=test_df,
                query_column="question",
                relevant_column=f"{summarization_type}_doc_ids",  # Use the updated "{summarization_type}_doc_ids" column
                k=k,
            )

            # Store the results
            all_metrics[summarization_type][f"k={k}"] = metrics
            print(f"Results for {summarization_type} with k={k}:\n")
            for metric, value in metrics.items():
                print(f"{metric}: {value}")

    # Save metrics to a JSON file
    with open(output_metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=4)

    print(f"\nMetrics saved to {output_metrics_path}")


if __name__ == "__main__":
    # Dataset paths
    # dataset = "COVIDQA"
    dataset = "emanual"
    dataset_path_train = f"summarized_data/{dataset}_train.parquet"
    dataset_path_test = f"summarized_data/{dataset}_test.parquet"

    # List of summarization types
    summarization_types = [
        "raw_data",
        "abstractive_summary",
        "extractive_summary",
        "filtered_summary",
    ]

    # List of k values for evaluation
    k_values = [1, 3, 7, 15]

    metrics_output_path = f"summarized_data/{dataset}_metrics2.json"

    # Run the training and evaluation process
    train_and_evaluate(
        dataset_path_train,
        dataset_path_test,
        dataset,
        summarization_types,
        k_values,
        metrics_output_path,
    )
