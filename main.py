# main.py

import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from processing import process_document

# Initialize the FastAPI application
app = FastAPI(
    title="Formative AI Processing Service",
    description="An API to process PDF documents into a Knowledge Graph using an LLM.",
    version="1.0.0"
)

# Define the directory to temporarily store uploaded files
UPLOADS_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)

@app.get("/", tags=["Root"])
def read_root():
    """
    Root endpoint with a welcome message.
    """
    return {
        "message": "Welcome to the Formative AI API.",
        "documentation": "Please visit /docs to see the API documentation and test the endpoints."
    }

@app.post("/process-pdf/", tags=["Processing"])
async def process_pdf_endpoint(file: UploadFile = File(..., description="The PDF file to be processed.")):
    """
    Accepts a PDF file, processes it through the entire pipeline,
    and populates the Neo4j Knowledge Graph.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDFs are accepted.")
    
    # Define a safe file path for the temporary file
    file_path = os.path.join(UPLOADS_DIR, file.filename)
    
    try:
        # Save the uploaded file to the server's disk
        print(f"Receiving file: {file.filename}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print("File saved. Starting processing pipeline...")
        # Call the main orchestrator function from processing.py
        result = process_document(file_path)
        
        # Return a success response
        return {
            "filename": file.filename,
            "message": "File processed successfully and Knowledge Graph populated.",
            "details": result
        }
    except Exception as e:
        # If any error occurs during processing, return a server error
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        # Crucially, clean up by deleting the temporary file after processing
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Cleaned up temporary file: {file.filename}")