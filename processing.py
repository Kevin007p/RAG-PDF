from pathlib import Path
import fitz  # PyMuPDF


def extract_text_and_images(pdf_file):
    """Main function to extract text and images from a PDF."""
    # Create directories for output
    new_dir = Path("Documents") / Path(pdf_file).stem
    new_dir.mkdir(parents=True, exist_ok=True)
    
    text_file_path = new_dir / "text.txt"
    image_dir = new_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    
    # Call the text and image extraction functions
    extract_text(pdf_file, text_file_path)
    extract_images(pdf_file, image_dir)



def extract_text(pdf_file, text_file_path):
    """Extract text from a PDF and save it to a text file."""
    document = fitz.open(pdf_file)
    
    with open(text_file_path, "w") as output:  # Use "w" to overwrite
        for page_num in range(len(document)):
            page = document[page_num]
            output.write(page.get_text())
            output.write("\n")
            output.write("###########")
            output.write("\n\n")
    
    print(f"Text successfully extracted and saved to {text_file_path}")


def extract_images(pdf_file, image_dir):
    """Extract images from a PDF and save them to a directory."""
    document = fitz.open(pdf_file)
    
    for page_num in range(len(document)):
        page = document[page_num]
        image_list = page.get_images(full=True)

        if image_list:
            print(f"Found {len(image_list)} images on page {page_num}")
        else:
            print(f"Did not find any images on page {page_num}")
        
        for image_index, img in enumerate(image_list):
            xref = img[0]
            
            # Extract image bytes
            base_image = document.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Get the image extension
            image_ext = base_image["ext"]
            
            # Save the image
            image_name = f"image{page_num+1}_{image_index}.{image_ext}"
            image_path = image_dir / image_name
            with open(image_path, "wb") as image_file:
                image_file.write(image_bytes)
                print(f"[+] Image saved as {image_path}")


def extract_text_and_images(pdf_file):
    """Main function to extract text and images from a PDF."""
    # Create directories for output
    new_dir = Path("Documents") / Path(pdf_file).stem
    new_dir.mkdir(parents=True, exist_ok=True)
    
    text_file_path = new_dir / "text.txt"
    image_dir = new_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    
    # Call the text and image extraction functions
    extract_text(pdf_file, text_file_path)
    extract_images(pdf_file, image_dir)
    
