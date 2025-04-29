import os
import json
import numpy as np
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Qdrant
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
import openai
import requests

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:12434/engines/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "ignored")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "ai/gemma3")

openai_client = openai.OpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY,
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

# Check for uploaded documents and add them to their collection
uploaded_files = [os.path.join(UPLOADS_PATH, f) for f in os.listdir(UPLOADS_PATH) if f.endswith('.pdf')]
if uploaded_files:
    uploaded_documents = []
    for pdf in uploaded_files:
        loader = PyPDFLoader(pdf)
        uploaded_documents.extend(loader.load())
    
    uploaded_chunks = splitter.split_documents(uploaded_documents)
    uploaded_embeddings = [model.encode(chunk.page_content) for chunk in uploaded_chunks]
    uploaded_payloads = [{"text": chunk.page_content, "source": os.path.basename(chunk.metadata.get('source', 'uploaded_document'))} for chunk in uploaded_chunks]
    
    qdrant_client.upsert(
        collection_name=UPLOADS_COLLECTION_NAME,
        points=[
            {
                "id": i,
                "vector": uploaded_embeddings[i],
                "payload": uploaded_payloads[i],
            }
            for i in range(len(uploaded_embeddings))
        ],
    )
    print(f"Loaded {len(uploaded_files)} uploaded document(s) with {len(uploaded_chunks)} chunks")

if __name__ == "__main__":
    print("BanglaChatbot is ready! Type 'exit' to quit.")
    chat_history = []
    while True:
        query = input("You: ")
        if query.lower() == 'exit':
            break
        
        # Embed the question for similarity search
        question_vec = model.encode(query)
        
        # Store the current question in chat context
        chat_context.add_question(query, question_vec)
        
        # Get similar previous questions
        similar_questions = chat_context.get_similar_questions(question_vec)
        
        # Search both Qdrant collections for document context
        original_hits = qdrant_client.search(
            collection_name=ORIGINAL_COLLECTION_NAME,
            query_vector=question_vec,
            limit=3,
        )
        
        upload_hits = qdrant_client.search(
            collection_name=UPLOADS_COLLECTION_NAME,
            query_vector=question_vec,
            limit=3,
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
            {"role": "system", "content": """You are an advanced Procurement Assistant named BRACGTP. You analyze documents, RFPs, contracts, and other procurement-related materials to provide accurate information and guidance.

INSTRUCTIONS:
1. Use ONLY the provided context to answer questions.
2. Format your responses in a clean, structured manner using HTML tags when appropriate.
3. Use <h2> for section headings, <ul> or <ol> for lists, <strong> for emphasis, and <p> for paragraphs.
4. When discussing financial data, use <table> tags with proper formatting.
5. If referencing specific sections from documents, clearly cite the source with page numbers if available.
6. If the answer cannot be derived from the context, clearly state this rather than making up information.
7. Be concise but thorough in your explanations.
8. If the user's question is similar to a previous question, acknowledge this and build upon previous responses.
9. IMPORTANT: Clearly distinguish between information from original documents vs. recently uploaded files.
10. When referencing recently uploaded documents, mention that they were uploaded by the user.

This formatting will ensure your responses display properly in HTML interfaces."""},
            {"role": "user", "content": f"Context: {combined_context}\n\nQuestion: {query}"}
        ]
        
        data = {
            "model": OPENAI_MODEL,
            "messages": messages,
            "max_tokens": 25600,
            "temperature": 0.7,
            "n": 1,
            "stream": True
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        with requests.post(OPENAI_BASE_URL + "/chat/completions", headers=headers, json=data, stream=True) as resp:
            print("Bot:", end=" ", flush=True)
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
                            print(content, end="", flush=True)
                    except Exception:
                        continue
            print()
        chat_history.append((query, "[streamed above]"))