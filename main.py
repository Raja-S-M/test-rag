from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from document_parser import parse_document
from rag_engine import index_document, query_rag

load_dotenv()

app = FastAPI(title="NotebookLLM Clone Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        text = parse_document(file.filename, contents)

        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="No text extracted.")

        index_document(file.filename, text)

        return {
            "status": "success",
            "filename": file.filename,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat(request: QueryRequest):
    try:
        return query_rag(request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents")
async def get_documents():
    return {"status": "ok", "docs": []}


@app.get("/api/health")
def health_check():
    return {"status": "ok"}