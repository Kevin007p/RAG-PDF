import json
from pathlib import Path
from sentence_transformers import SentenceTransformer

def create_embeddings(input_dir, output_file, model_path, batch_size=16):
    """
    Create embeddings from text files in the input directory using a specified model.

    Args:
        input_dir (str): Directory containing lecture subdirectories with `text.txt` files.
        output_file (str): Path to save the JSON file with embeddings.
        model_path (str): Path to the embedding model (pretrained or fine-tuned).
        batch_size (int): Number of chunks to embed in a single batch.
    """
    # Load the specified model
    model = SentenceTransformer(model_path)
    input_path = Path(input_dir)
    all_data = []

    for lecture_dir in input_path.glob("*"):
        if lecture_dir.is_dir():
            text_file = lecture_dir / "text.txt"
            if not text_file.exists():
                print(f"Warning: Missing text.txt in {lecture_dir}")
                continue
            
            with open(text_file, "r", encoding="utf-8") as file:
                text = file.read()

            # Split text into chunks
            chunks = text.split("###########")
            processed_chunks = [
                f"Lecture: {lecture_dir.stem}. Content: {chunk.strip()}" 
                for chunk in chunks 
                if chunk.strip()
            ]

            # Create embeddings in batches
            for i in range(0, len(processed_chunks), batch_size):
                batch = processed_chunks[i:i+batch_size]
                embeddings = model.encode(batch)

                for j, embedding in enumerate(embeddings):
                    all_data.append({
                        "content": processed_chunks[i + j],
                        "embedding": embedding.tolist(),
                        "metadata": {
                            "lecture": lecture_dir.stem,
                            "chunk": i + j + 1
                        }
                    })
    
    # Save the prepared data to a JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f)

    print(f"Embeddings and metadata saved to {output_file}")

if __name__ == "__main__":
    # Create embeddings for the pretrained model
    create_embeddings(
        input_dir="Documents",
        output_file="prepared_data_pretrained.json",
        model_path="multi-qa-mpnet-base-dot-v1"
    )
    
    # Create embeddings for the fine-tuned model
    create_embeddings(
        input_dir="Documents",
        output_file="prepared_data_finetuned.json",
        model_path="fine_tuned_model"
    )
