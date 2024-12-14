from pathlib import Path
import fitz  # PyMuPDF

def extract_text_from_pdfs(input_dir, output_dir, chunk_size=5):
    """
    Extract text from PDF files, chunk it into groups of lines, and save to text files.
    
    Args:
        input_dir (str): Directory containing input PDFs.
        output_dir (str): Directory to save extracted text files.
        chunk_size (int): Number of lines per chunk. Default is 5.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for pdf_file in input_path.glob("*.pdf"):
        document = fitz.open(pdf_file)
        lecture_output_dir = output_path / pdf_file.stem
        lecture_output_dir.mkdir(parents=True, exist_ok=True)
        text_file_path = lecture_output_dir / "text.txt"

        with open(text_file_path, "w", encoding="utf-8") as output_file:
            chunk_lines = []  # Temporary storage for lines in a chunk
            for page_num in range(len(document)):
                page = document[page_num]
                lines = page.get_text().splitlines()
                
                for line in lines:
                    if line.strip():  # Skip empty lines
                        chunk_lines.append(line.strip())

                    # If we reach the chunk size, write the chunk
                    if len(chunk_lines) >= chunk_size:
                        output_file.write("\n".join(chunk_lines))
                        output_file.write("\n###########\n\n")
                        chunk_lines = []

            # Write any remaining lines in the last chunk
            if chunk_lines:
                output_file.write("\n".join(chunk_lines))
                output_file.write("\n###########\n\n")

        print(f"Extracted text for {pdf_file.stem} into {text_file_path}")

if __name__ == "__main__":
    # Adjust chunk_size as needed (e.g., based on average slide content)
    extract_text_from_pdfs("lectures", "Documents", chunk_size=5)
