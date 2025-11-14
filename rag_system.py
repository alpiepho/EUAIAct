import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
import fitz
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

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

# Data source configuration
WEB_LINK = os.getenv('WEB_LINK', '')
PDF_FILENAME = os.getenv('PDF_FILENAME', 'EU_AI_Act.pdf')

# Determine data source priority: WEB_LINK takes precedence over PDF_FILENAME
USE_WEB = bool(WEB_LINK)
DATA_SOURCE = WEB_LINK if USE_WEB else PDF_FILENAME

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

def fetch_web_content(url):
    """
    Fetch and extract text content from a web page.
    Returns a list of text chunks from the page.
    """
    import time
    
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            print(f"Fetching content from: {url} (attempt {attempt + 1}/{max_retries})")
            
            # More comprehensive headers to appear as a real browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Cache-Control': 'max-age=0'
            }
            
            # Create a session for better connection handling
            session = requests.Session()
            session.headers.update(headers)
            
            response = session.get(url, timeout=30, verify=True, allow_redirects=True)
            response.raise_for_status()
            
            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script, style, and navigation elements
            for element in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
                element.decompose()
            
            # Try to find main content area first
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content') or soup.body
            
            if main_content:
                # Extract text from main content
                text = main_content.get_text(separator='\n', strip=True)
            else:
                # Fallback to full page text
                text = soup.get_text(separator='\n', strip=True)
            
            # Clean up text: remove extra whitespace and empty lines
            lines = [line.strip() for line in text.split('\n') if line.strip() and len(line.strip()) > 10]
            text = '\n'.join(lines)
            
            if len(text) < 100:
                raise ValueError(f"Extracted text too short ({len(text)} chars), might be blocked or empty page")
            
            # Split into chunks (approximately 1000 characters each)
            chunk_size = 1000
            chunks = []
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i + chunk_size]
                if chunk.strip():
                    chunks.append(chunk)
            
            print(f"✓ Successfully fetched {len(chunks)} chunks ({len(text)} total chars) from web page")
            session.close()
            return chunks
            
        except requests.exceptions.ConnectionError as e:
            print(f"⚠ Connection error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                print(f"  Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"✗ Failed to fetch content after {max_retries} attempts")
                return []
                
        except Exception as e:
            print(f"✗ Error fetching web content: {e}")
            if attempt < max_retries - 1:
                print(f"  Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                return []
    
    return []

def get_or_create_collection():
    collection_name = "rag_collection"
    collection = None
    needs_loading = False
    
    try:
        # Try to get existing collection
        collection = chroma_client.get_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
        doc_count = collection.count()
        
        if doc_count == 0:
            print(f"⚠ Collection exists but is empty, will reload data")
            # Delete and recreate the collection
            chroma_client.delete_collection(name=collection_name)
            collection = chroma_client.create_collection(
                name=collection_name,
                embedding_function=embedding_function
            )
            needs_loading = True
        else:
            print(f"✓ Using existing collection with {doc_count} documents")
            
    except:
        # Create new collection if it doesn't exist
        print(f"Creating new collection")
        collection = chroma_client.create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
        needs_loading = True
    
    # Load data if needed
    if needs_loading:
        # Load data based on source type
        if USE_WEB:
            # Fetch and process web content
            print(f"Loading data from web: {WEB_LINK}")
            chunks = fetch_web_content(WEB_LINK)
            
            if chunks:
                try:
                    for i, chunk in enumerate(chunks):
                        collection.add(
                            documents=[chunk],
                            ids=[f"web_chunk_{i}"],
                            metadatas=[{"chunk": i, "source": WEB_LINK, "type": "web"}]
                        )
                    print(f"✓ Added {len(chunks)} web chunks to collection")
                except Exception as e:
                    print(f"Error adding web content to collection: {e}")
            else:
                print("⚠ Warning: No content fetched from web page")
        else:
            # Load PDF and add to collection
            print(f"Loading data from PDF: {PDF_FILENAME}")
            try:
                doc = fitz.open(PDF_FILENAME)
                for i, page in enumerate(doc):
                    text = page.get_text()
                    if text.strip():  # Only add non-empty pages
                        collection.add(
                            documents=[text],
                            ids=[f"page_{i}"],
                            metadatas=[{"page": i, "source": PDF_FILENAME, "type": "pdf"}]
                        )
                print(f"✓ Added {len(doc)} pages from PDF to collection")
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
    Generate dynamic UI content based on the data source (web or PDF) and document content.
    Returns a dictionary with all UI strings for the Streamlit app.
    """
    # Extract document/page name from source
    if USE_WEB:
        # Extract domain or page title from URL
        parsed_url = urlparse(WEB_LINK)
        doc_name = parsed_url.netloc.replace('www.', '').replace('.', ' ').title()
        source_type = "website"
    else:
        # Extract name from PDF filename
        doc_name = PDF_FILENAME.replace('.pdf', '').replace('_', ' ').replace('-', ' ')
        source_type = "document"
    
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
        "source_display": WEB_LINK if USE_WEB else PDF_FILENAME,
        "source_type": source_type
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