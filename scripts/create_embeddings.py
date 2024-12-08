import json
from pathlib import Path
from sentence_transformers import SentenceTransformer

def create_embeddings(input_dir, output_file="prepared_data.json", batch_size=16):
    """
    Create embeddings from text files in the input directory.

    Args:
        input_dir (str): Directory containing lecture subdirectories with `text.txt` files.
        output_file (str): Path to save the JSON file with embeddings.
        batch_size (int): Number of chunks to embed in a single batch.
    """
    model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
    input_path = Path(input_dir)
    all_data = []

    for lecture_dir in input_path.glob("*"):
        if lecture_dir.is_dir():
            text_file = lecture_dir / "text.txt"
            if not text_file.exists():
                print(f"Warning: Missing text.txt in {lecture_dir}")
                continue
            
            with open(text_file, "r") as file:
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
    with open(output_file, "w") as f:
        json.dump(all_data, f)

    print(f"Embeddings and metadata saved to {output_file}")

if __name__ == "__main__":
    create_embeddings("Documents")
