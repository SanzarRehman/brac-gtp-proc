from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
import shutil
from pathlib import Path
import json
import numpy as np
import ollama
import requests
import openai
from transformers import pipeline

from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Qdrant
from langchain_community.llms import Ollama
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
import os
from datetime import datetime

app = FastAPI()

class ChatRequest(BaseModel):
    question: str
    chat_history: list = []

# Define paths for original and uploaded documents
DOCS_PATH = os.path.join(os.path.dirname(__file__), 'doccuments')
UPLOADS_PATH = os.path.join(os.path.dirname(__file__), 'uploads')

# Create uploads folder if it doesn't exist
if not os.path.exists(UPLOADS_PATH):
    os.makedirs(UPLOADS_PATH)

# Define separate collection names
ORIGINAL_COLLECTION_NAME = "bangla_chatbot_docs"
UPLOADS_COLLECTION_NAME = "bangla_chatbot_uploads"

# Setup for original documents
pdf_files = [os.path.join(DOCS_PATH, f) for f in os.listdir(DOCS_PATH) if f.endswith('.pdf')]
documents = []
for pdf in pdf_files:
    loader = PyPDFLoader(pdf)
    documents.extend(loader.load())
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = [model.encode(chunk.page_content) for chunk in chunks]
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

# Setup Qdrant client
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Initialize original documents collection
try:
    qdrant_client.delete_collection(collection_name=ORIGINAL_COLLECTION_NAME)
except Exception as e:
    print(f"Original collection delete skipped or failed: {e}")
try:
    qdrant_client.create_collection(
        collection_name=ORIGINAL_COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
except Exception as e:
    print(f"Original collection creation skipped or failed: {e}")

# Initialize uploads collection if it doesn't exist
try:
    if not qdrant_client.collection_exists(UPLOADS_COLLECTION_NAME):
        qdrant_client.create_collection(
            collection_name=UPLOADS_COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
except Exception as e:
    print(f"Uploads collection creation skipped or failed: {e}")

# Insert original documents into their collection
payloads = [{"text": chunk.page_content, "source": "original"} for chunk in chunks]
if payloads:  # Only insert if there are documents
    qdrant_client.upsert(
        collection_name=ORIGINAL_COLLECTION_NAME,
        points=[
            {
                "id": i,
                "vector": embeddings[i],
                "payload": payloads[i],
            }
            for i in range(len(embeddings))
        ],
    )

# Remove Ollama and use Hugging Face pipeline for LLM
llm_pipeline = pipeline("text-generation", model="gpt2")  # You can change to any open LLM
GEMMA_API_URL = "http://localhost:8000/v1/chat/completions"  # Update if your endpoint is different
GEMMA_API_KEY = os.getenv("GEMMA_API_KEY", "")  # Set in your environment if needed

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:12434/engines/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "ignored")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "ai/gemma3")

openai_client = openai.OpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY,
)

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    def stream_response():
        # Embed the question for similarity search
        question_vec = model.encode(request.question)
        
        # Store the current question in chat context
        chat_context.add_question(request.question, question_vec)
        
        # Get similar previous questions
        similar_questions = chat_context.get_similar_questions(question_vec)
        
        # Search both Qdrant collections for document context
        original_hits = qdrant_client.search(
            collection_name=ORIGINAL_COLLECTION_NAME,
            query_vector=question_vec,
            limit=5,
        )
        
        upload_hits = qdrant_client.search(
            collection_name=UPLOADS_COLLECTION_NAME,
            query_vector=question_vec,
            limit=5,
        )
        
        # Format document context with source indications
        document_context = ""
        
        if original_hits:
            document_context += "\nFROM BASE DOCUMENTS:\n"
            for hit in original_hits:
                document_context += f"Source: Original documentation\nContent: {hit.payload['text']}\n\n"
        
        if upload_hits:
            document_context += "\nFROM RECENTLY UPLOADED DOCUMENTS:\n"
            for hit in upload_hits:
                document_context += f"Source: {hit.payload['source']}\nContent: {hit.payload['text']}\n\n"
        
        # Get file contexts
        file_contexts = chat_context.get_file_contexts()
        file_context_str = ""
        if file_contexts:
            file_context_str = "\nRECENTLY UPLOADED FILES SUMMARY:\n"
            for ctx in file_contexts[-3:]:  # Use only the 3 most recent files
                file_context_str += f"Filename: {ctx.get('filename')}\nSummary: {ctx.get('summary')}\n\n"
        
        # Build previous conversation context
        conversation_context = ""
        if similar_questions:
            conversation_context = "\nRELATED PREVIOUS QUESTIONS:\n"
            for q, sim in similar_questions:
                conversation_context += f"- {q} (similarity: {sim:.2f})\n"
        
        # Combine all context
        combined_context = document_context + file_context_str + conversation_context
        
        # Prepare messages for API call
        messages = [
            {"role": "system", "content": """You are BRAC's Procurement Assistant named BRACGPT. You are an experienced procurement professional working for BRAC.

INSTRUCTIONS:
1. Respond as a knowledgeable procurement professional working for BRAC.
2. Format your responses in a clean, structured manner using HTML tags when appropriate.
3. Use <h2> for section headings, <ul> or <ol> for lists, <strong> for emphasis, and <p> for paragraphs.
4. When discussing financial data, use <table> tags with proper formatting.
5. If referencing specific sections from procurement documents, clearly cite the reference.
6. If you cannot provide an answer, politely state that you'll need to consult with the procurement team.
7. Be concise but thorough in your explanations.
8. If a user's question is similar to a previous question, acknowledge this and build upon previous responses.
9. Never reveal that you're using any default documents or stored knowledge base.
10. For uploaded documents, simply respond as if you've personally reviewed them as a procurement officer.

Your goal is to provide helpful procurement guidance while maintaining a professional BRAC organizational identity."""},
            {"role": "user", "content": f"Context: {combined_context}\n\nQuestion: {request.question}"}
        ]
        
        data = {
            "model": OPENAI_MODEL,
            "messages": messages,
            "max_tokens": 25600,
            "temperature": 0.5,
            "n": 1,
            "stream": True
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        with requests.post(OPENAI_BASE_URL + "/chat/completions", headers=headers, json=data, stream=True) as resp:
            for line in resp.iter_lines():
                if line:
                    if line.startswith(b'data: '):
                        line = line[6:]
                    if line == b'[DONE]':
                        break
                    try:
                        chunk = json.loads(line)
                        content = chunk.get('choices', [{}])[0].get('delta', {}).get('content', '')
                        if content:
                            yield content.encode('utf-8')  # Yield bytes for StreamingResponse
                    except Exception:
                        continue
    return StreamingResponse(stream_response(), media_type="text/plain")

@app.get("/")
def serve_frontend():
    return FileResponse("simple_chat.html")

@app.get("/openapi.json", include_in_schema=False)
def custom_openapi():
    return get_openapi(
        title="BanglaChatbot API",
        version="1.0.0",
        description="API for BanglaChatbot with PDF context and streaming responses using Open LLM.",
        routes=app.routes,
    )

# Add a class to store chat context with embedded files
class ChatContext:
    def __init__(self):
        self.previous_questions = []
        self.previous_embeddings = []
        self.file_contexts = {}  # Map of file_id to file content context
    
    def add_question(self, question, embedding):
        self.previous_questions.append(question)
        self.previous_embeddings.append(embedding)
        # Keep only the last 10 questions for context
        if len(self.previous_questions) > 10:
            self.previous_questions.pop(0)
            self.previous_embeddings.pop(0)
    
    def add_file_context(self, file_id, context):
        self.file_contexts[file_id] = context
    
    def get_similar_questions(self, query_embedding, top_k=3):
        if not self.previous_embeddings:
            return []
        
        # Calculate cosine similarity
        similarities = []
        for i, emb in enumerate(self.previous_embeddings):
            similarity = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
            similarities.append((i, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k similar questions
        return [(self.previous_questions[i], sim) for i, sim in similarities[:top_k] if sim > 0.7]
    
    def get_file_contexts(self):
        return list(self.file_contexts.values())

# Initialize chat context
chat_context = ChatContext()

# Modify the upload endpoint to store file context
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # Save uploaded file to the uploads folder
    file_path = os.path.join(UPLOADS_PATH, file.filename)
    
    try:
        # Create the destination folder if it doesn't exist
        if not os.path.exists(UPLOADS_PATH):
            os.makedirs(UPLOADS_PATH)
            
        # Write the file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Process and embed the new document
        new_loader = PyPDFLoader(file_path)
        new_documents = new_loader.load()
        new_chunks = splitter.split_documents(new_documents)
        
        # Generate embeddings for new chunks
        new_embeddings = [model.encode(chunk.page_content) for chunk in new_chunks]
        new_payloads = [{"text": chunk.page_content, "source": file.filename} for chunk in new_chunks]
        
        # Store file context in chat context
        file_id = f"file_{hash(file.filename)}_{len(chat_context.file_contexts)}"
        file_summary = "\n".join([chunk.page_content for chunk in new_chunks[:3]])  # Use first 3 chunks as summary
        chat_context.add_file_context(file_id, {
            "filename": file.filename,
            "summary": file_summary,
            "upload_time": str(datetime.now())
        })
        
        # Add to the uploads collection in Qdrant instead of the main collection
        start_id = qdrant_client.count(collection_name=UPLOADS_COLLECTION_NAME).count
        qdrant_client.upsert(
            collection_name=UPLOADS_COLLECTION_NAME,
            points=[
                {
                    "id": start_id + i,
                    "vector": new_embeddings[i],
                    "payload": new_payloads[i],
                }
                for i in range(len(new_embeddings))
            ],
        )
        
        return JSONResponse(content={
            "filename": file.filename,
            "status": "success",
            "chunks": len(new_chunks),
            "message": f"Document successfully uploaded and processed into {len(new_chunks)} chunks."
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "filename": file.filename,
                "status": "error",
                "message": f"Failed to process document: {str(e)}"
            }
        )

# Add endpoint to list available documents
@app.get("/documents")
async def list_documents():
    if not os.path.exists(DOCS_PATH):
        return {"documents": []}
        
    files = [f for f in os.listdir(DOCS_PATH) if f.endswith('.pdf')]
    return {"documents": files}

# Add endpoint to clear all vectors from the collection
@app.post("/clear-all")
async def clear_all_data():
    try:
        # Delete and recreate both collections
        for collection_name in [ORIGINAL_COLLECTION_NAME, UPLOADS_COLLECTION_NAME]:
            try:
                # Delete if exists
                if qdrant_client.collection_exists(collection_name):
                    qdrant_client.delete_collection(collection_name=collection_name)
                
                # Recreate the collection
                qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                )
            except Exception as e:
                print(f"Collection {collection_name} operation failed: {e}")
        
        # Reset the chat context
        global chat_context
        chat_context = ChatContext()
        
        # Re-initialize original documents if needed
        if pdf_files:
            # Reinsert original documents into their collection
            payloads = [{"text": chunk.page_content, "source": "original"} for chunk in chunks]
            if payloads:
                qdrant_client.upsert(
                    collection_name=ORIGINAL_COLLECTION_NAME,
                    points=[
                        {
                            "id": i,
                            "vector": embeddings[i],
                            "payload": payloads[i],
                        }
                        for i in range(len(embeddings))
                    ],
                )
        
        return JSONResponse(content={
            "status": "success",
            "message": "Successfully cleared all vector databases and chat context."
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Failed to clear data: {str(e)}"
            }
        )