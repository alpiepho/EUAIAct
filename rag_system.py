import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
import fitz

# Load environment variables (override=True means .env file takes precedence)
load_dotenv(override=True)

# Check if OpenAI API key is set and non-default
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
USE_OPENAI = OPENAI_API_KEY and OPENAI_API_KEY != 'default' and len(OPENAI_API_KEY) > 10

# Ollama configuration
OLLAMA_HOST = os.getenv('OLLAMA_HOST', '')
# Ensure OLLAMA_HOST has a protocol
if OLLAMA_HOST and not OLLAMA_HOST.startswith(('http://', 'https://')):
    OLLAMA_HOST = f"http://{OLLAMA_HOST}"
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2')

# PDF configuration
PDF_FILENAME = os.getenv('PDF_FILENAME', 'EU_AI_Act.pdf')

# Initialize client based on configuration
if USE_OPENAI:
    print("Using OpenAI API")
    client = OpenAI(api_key=OPENAI_API_KEY)
    MODEL_NAME = "gpt-4o-mini"
else:
    if not OLLAMA_HOST:
        raise ValueError(
            "OLLAMA_HOST is not set in environment variables. "
            "Please set OLLAMA_HOST in your .env file (e.g., OLLAMA_HOST=http://10.0.0.60:11434) "
            "or set OPENAI_API_KEY to use OpenAI instead."
        )
    print(f"Using Ollama at {OLLAMA_HOST}")
    client = OpenAI(
        base_url=f"{OLLAMA_HOST}/v1",
        api_key="ollama"  # Ollama doesn't require a real API key
    )
    MODEL_NAME = OLLAMA_MODEL
    
    # Test Ollama connection and verify model availability
    try:
        import urllib.request
        import json
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags") as response:
            data = json.loads(response.read().decode())
            models = data.get('models', [])
            model_names = [m['name'] for m in models]
            if any(OLLAMA_MODEL in name for name in model_names):
                print(f"✓ Ollama connection successful - model '{OLLAMA_MODEL}' is available")
            else:
                print(f"⚠ Warning: Model '{OLLAMA_MODEL}' not found. Available models: {', '.join(model_names)}")
    except Exception as e:
        print(f"⚠ Warning: Could not connect to Ollama - {str(e)}")

# Initialize ChromaDB
chroma_client = chromadb.Client()

# Initialize embedding function based on configuration
if USE_OPENAI:
    embedding_function = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-embedding-3-small"
    )
else:
    # Using a simple sentence transformer for Ollama
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

def get_or_create_collection():
    collection_name = "eu_ai_act"
    try:
        # Try to get existing collection
        collection = chroma_client.get_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
    except:
        # Create new collection if it doesn't exist
        collection = chroma_client.create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
        
        # Load PDF and add to collection
        try:
            doc = fitz.open(PDF_FILENAME)
            for i, page in enumerate(doc):
                text = page.get_text()
                if text.strip():  # Only add non-empty pages
                    collection.add(
                        documents=[text],
                        ids=[f"page_{i}"],
                        metadatas=[{"page": i, "source": PDF_FILENAME}]
                    )
        except Exception as e:
            print(f"Error loading PDF: {e}")
    
    return collection

def get_relevant_sections(query):
    collection = get_or_create_collection()
    results = collection.query(
        query_texts=[query],
        n_results=3
    )
    return results['documents'][0]

def generate_answer(query, return_sources=False):
    try:
        # Get collection
        collection = get_or_create_collection()
        
        # Query collection
        results = collection.query(
            query_texts=[query],
            n_results=3
        )
        
        # Prepare context from results
        relevant_texts = results['documents'][0]
        context = "\n".join(relevant_texts)
        
        # Generate response using configured model
        messages = [
            {"role": "system", "content": """You are an AI assistant specialized in the EU AI Act. 
             Provide accurate, clear, and concise answers based on the provided context. 
             When referring to specific parts of the Act, mention this explicitly in your response."""},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nPlease provide a clear answer based on the context, mentioning relevant articles or sections when applicable."}
        ]
        
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.7
        )
        
        if return_sources:
            return completion.choices[0].message, relevant_texts
        else:
            return completion.choices[0].message
    
    except Exception as e:
        raise Exception(f"Error generating answer: {str(e)}")

def format_source_reference(text):
    """Format the source text for display"""
    # You can add additional formatting logic here
    return text.strip()

def generate_ui_content():
    """
    Generate dynamic UI content based on the PDF filename and document content.
    Returns a dictionary with all UI strings for the Streamlit app.
    """
    # Extract document name from filename
    doc_name = PDF_FILENAME.replace('.pdf', '').replace('_', ' ').replace('-', ' ')
    
    # Try to generate context-aware content using RAG
    try:
        collection = get_or_create_collection()
        
        # Query for document overview
        overview_results = collection.query(
            query_texts=["What is this document about? Main topics and purpose."],
            n_results=2
        )
        # overview_results['documents'][0] is a list of strings (one per result)
        # Join them and then take first 50 words
        overview_full_text = " ".join(overview_results['documents'][0])
        overview_words = overview_full_text.split()[:50]
        overview_text = " ".join(overview_words) + "..."
        
        # Generate example questions using AI
        context = "\n".join(overview_results['documents'][0])
        messages = [
            {"role": "system", "content": "You are a helpful assistant that generates relevant questions about documents."},
            {"role": "user", "content": f"Based on this document content:\n{context[:1500]}\n\nGenerate 4 specific, diverse example questions that users might ask. Return ONLY the questions, one per line, without numbers or bullets."}
        ]
        
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.7,
            max_tokens=200
        )
        
        generated_questions = [q.strip() for q in completion.choices[0].message.content.strip().split('\n') if q.strip()]
        example_questions = generated_questions[:4] if len(generated_questions) >= 4 else [
            f"What is the main purpose of {doc_name}?",
            f"What are the key topics covered in {doc_name}?",
            f"What are the main requirements in {doc_name}?",
            f"Who does {doc_name} apply to?"
        ]
        
    except Exception as e:
        print(f"Could not generate dynamic content: {e}")
        # Fallback to generic content
        overview_text = f"This document contains information about {doc_name}."
        example_questions = [
            f"What is {doc_name}?",
            f"What are the key topics in {doc_name}?",
            f"What are the main requirements?",
            f"Who does this apply to?"
        ]
    
    return {
        "title": f"{doc_name} Chat Assistant",
        "subtitle": f"Ask any question about {doc_name}, and chat with the AI assistant!",
        "chat_placeholder": f"What would you like to know about {doc_name}?",
        "about_text": f"This is an AI assistant specialized in answering questions about {doc_name}.",
        "example_questions": example_questions,
        "references_info": f"Ask a question to see relevant references from {doc_name}",
        "doc_name": doc_name,
        "doc_overview": overview_text,
        "pdf_filename": PDF_FILENAME
    }

if __name__ == "__main__":
    # Test the system
    query = "What is the EU AI Act?"
    answer, sources = generate_answer(query, return_sources=True)
    print("Answer:", answer.content)
    print("\nSources:")
    for i, source in enumerate(sources, 1):
        print(f"\nSource {i}:")
        print(format_source_reference(source))