from fastapi import FastAPI, UploadFile
from pathlib import Path


app = FastAPI()

@app.get("/")
def root():
    return {"Hello": "World"}

@app.post("/upload")
async def upload_file(uploaded_file: UploadFile):
    if (uploaded_file.filename.endswith(".pdf")):
        # Save file
        
        path = Path(f"RawPDFs/{uploaded_file.filename}")
        path.mkdir(parents=True, exist_ok=True)
        return {"message": f"File uploaded succesfully and saved to {path}"}
    else:
        return {"error": "Sorry, only PDF files are accepted"}
        