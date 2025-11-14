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
            doc = fitz.open("EU_AI_Act.pdf")
            for i, page in enumerate(doc):
                text = page.get_text()
                if text.strip():  # Only add non-empty pages
                    collection.add(
                        documents=[text],
                        ids=[f"page_{i}"],
                        metadatas=[{"page": i, "source": "EU_AI_Act.pdf"}]
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

if __name__ == "__main__":
    # Test the system
    query = "What is the EU AI Act?"
    answer, sources = generate_answer(query, return_sources=True)
    print("Answer:", answer.content)
    print("\nSources:")
    for i, source in enumerate(sources, 1):
        print(f"\nSource {i}:")
        print(format_source_reference(source))