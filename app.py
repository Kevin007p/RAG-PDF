from fastapi import FastAPI, UploadFile
from pathlib import Path


app = FastAPI()

@app.get("/")
def root():
    return {"Hello": "World"}

@app.post("/upload")
async def upload_file(uploaded_file: UploadFile):
    if (uploaded_file.filename.endswith(".pdf")):
        
        pdfs = Path("RawPDFs")
        pdfs.mkdir(parents=True, exist_ok=True)
        file_path = pdfs / uploaded_file.filename

        with open(file_path, "wb") as f:
            f.write(await uploaded_file.read())
        
        return {"message": f"File uploaded succesfully and saved to {file_path}"}
    else:
        return {"error": "Sorry, only PDF files are accepted"}
        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)