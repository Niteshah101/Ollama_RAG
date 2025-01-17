import gradio as gr
import requests
import json
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import hashlib  # For hashing file content

# Directory to store vector stores
VECTOR_STORE_DIR = "vector_stores"
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# Initialize the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(pdf_file.name)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

# Function to compute hash of a file's content
def compute_file_hash(file_path):
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

# Function to create and store vector embeddings
def create_vector_store(pdf_text, hash_key):
    # Split the PDF text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(pdf_text)
    
    # Create embeddings and store in FAISS vector store
    vector_store = FAISS.from_texts(chunks, embedding_model)
    vector_store_path = os.path.join(VECTOR_STORE_DIR, f"{hash_key}.faiss")
    vector_store.save_local(vector_store_path)
    print(f"Vector store created and saved for hash: {hash_key}")
    return vector_store_path

# Function to load the vector store
def load_vector_store(hash_key):
    vector_store_path = os.path.join(VECTOR_STORE_DIR, f"{hash_key}.faiss")
    return FAISS.load_local(vector_store_path, embedding_model, allow_dangerous_deserialization=True)

# Function to query Ollama for a response
def query_ollama(prompt):
    """Send a prompt to the Ollama model and return the response."""
    url = "http://localhost:11434/api/generate"  # Ollama API endpoint
    payload = {
        "model": "llama3.2",  # Ensure the correct model name
        "prompt": prompt,
        "stream": False
    }
    headers = {'Content-Type': 'application/json'}
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Raise an exception for bad status codes
        answer = response.json().get("response", "No answer available")
        return answer
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return "Error"

# Main function to handle user questions and optional PDF input
def ask_question(question, pdf_file=None):
    vector_store = None
    if pdf_file:
        # Compute hash of the file
        file_hash = compute_file_hash(pdf_file.name)
        vector_store_path = os.path.join(VECTOR_STORE_DIR, f"{file_hash}.faiss")
        
        # Check if vector store already exists
        if os.path.exists(vector_store_path):
            print(f"Using cached vector store for file: {pdf_file.name}")
            vector_store = load_vector_store(file_hash)
        else:
            print(f"Creating new vector store for file: {pdf_file.name}")
            pdf_text = extract_text_from_pdf(pdf_file)
            create_vector_store(pdf_text, file_hash)
            vector_store = load_vector_store(file_hash)
    else:
        print("No file provided, please upload a PDF.")
        return "Please upload a PDF to provide context."

    # Retrieve relevant chunks from the vector store
    docs = vector_store.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content.strip() for doc in docs])
    
    # Create the augmented prompt for Ollama
    prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
    
    # Query Ollama with the augmented prompt
    return query_ollama(prompt)

# Gradio interface for the RAG-based Q&A system
interface = gr.Interface(
    fn=ask_question,
    inputs=[gr.Textbox(label="Ask Ollama"), gr.File(label="Upload PDF (Optional)")],
    outputs=gr.Textbox(label="Ollama's Answer"),
    title="Ollama Q&A with PDF Support (RAG Enabled)",
    description="Ask questions and get answers from Ollama using RAG. Optionally, upload a PDF to provide context."
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=1111, share=True)
