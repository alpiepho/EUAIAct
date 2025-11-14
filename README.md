# EU AI Act Chat Assistant

This project provides an interactive Streamlit-based chat assistant designed to answer questions about the EU AI Act. It leverages a retrieval-augmented generation (RAG) system to fetch and present relevant information from the EU AI Act documents, supporting both OpenAI and local Ollama models for flexibility.

![EU AI Act Chat Assistant](assets/euaiact-rag-chat.png)

## Features
- **Interactive Chat Interface**: Users can ask questions about the EU AI Act and receive real-time answers from the assistant.
- **Flexible Model Support**: Use either OpenAI's models or a local Ollama instance based on your configuration.
- **Reference Retrieval**: Provides references to relevant sections of the EU AI Act document in response to user queries.
- **Clear Chat History**: Users can reset their chat history at any time.
- **Sample Questions**: The sidebar includes sample questions for quick reference.
- **Document Overview**: Summarizes available documents in the sidebar for easy access.

---

## Project Structure

- **streamlit_app.py**: Defines the chat UI in Streamlit, handles chat history, and displays references.
- **rag_system.py**: Manages the RAG system by embedding and storing the EU AI Act document and querying relevant sections for answers.

---

## Installation

### Requirements
- Python 3.7+
- **Optional**: OpenAI API key for using OpenAI models
- **Optional**: Ollama instance for using local models (defaults to Ollama if no OpenAI key is provided)

### Setting Up the Virtual Environment

#### Windows
```bash
# Clone the repository and navigate to the project directory
git clone https://github.com/ingridstevens/EUAIAct
cd EUAIAct

# Create a virtual environment
python -m venv env

# Activate the virtual environment
env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### macOS / Linux
```bash
# Clone the repository and navigate to the project directory
git clone https://github.com/ingridstevens/EUAIAct
cd EUAIAct

# Create a virtual environment
python3 -m venv env

# Activate the virtual environment
source env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

#### Option 1: Using OpenAI
If you have an OpenAI API key, set it in your `.env` file:
```plaintext
OPENAI_API_KEY=your_openai_api_key_here
```

#### Option 2: Using Ollama (Default)
If you don't set an OpenAI API key (or set it to 'default'), the app will automatically use Ollama. Configure your Ollama settings in `.env`:
```plaintext
# Leave OPENAI_API_KEY unset or commented out to use Ollama
# OPENAI_API_KEY=

# Ollama host (default: http://10.0.0.60:11434)
OLLAMA_HOST=http://10.0.0.60:11434

# Ollama model (default: llama3.2)
# Options: llama3.2, mistral, qwen2.5, etc.
OLLAMA_MODEL=llama3.2
```

**Note**: Make sure your Ollama instance has the model pulled:
```bash
ollama pull llama3.2
```

#### Option 3: Using a Custom PDF Document
The app can work with any PDF document, not just the EU AI Act. The UI will dynamically adapt to your document:

```plaintext
# Set the PDF filename in your .env file
PDF_FILENAME=YourDocument.pdf
```

**Features when using a custom PDF:**
- **Dynamic Title**: The app title automatically updates based on your PDF filename
- **AI-Generated Questions**: Example questions are generated from your document's actual content
- **Context-Aware UI**: All UI elements (descriptions, placeholders, etc.) adapt to your document
- **Document Overview**: Automatically generated summary from your PDF content

Simply place your PDF in the project root directory and set the `PDF_FILENAME` variable. The app will automatically:
1. Extract and embed the PDF content
2. Generate relevant example questions using AI
3. Adapt all UI text to match your document's context

---

## How to Run the Project

### Start the Chat Application
1. Activate your virtual environment:
   - **Windows**: `env\Scripts\activate`
   - **macOS/Linux**: `source env/bin/activate`
2. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```
3. Open the link provided in the terminal (usually [http://localhost:8501](http://localhost:8501)) to interact with the assistant.

---

## How It Works

### Chat Interface (`streamlit_app.py`)
- **Main Interface**: Displays the title and a brief prompt for users to ask questions.
- **Dynamic UI Generation**: Automatically adapts all UI elements based on the configured PDF document.
- **Message History**: Shows previous messages from the user and assistant, storing chat history in session state.
- **Sidebar**: Contains options to clear chat history, view example questions, and reference current documents.

### RAG System (`rag_system.py`)
- **Automatic Model Selection**: Automatically uses OpenAI if an API key is configured, otherwise defaults to Ollama.
- **Document Embedding and Search**: The configured PDF document is embedded and stored in ChromaDB. Uses OpenAI embeddings when using OpenAI, or local sentence-transformer embeddings when using Ollama.
- **Answer Generation**: Constructs a response based on relevant document excerpts and the question, using either OpenAI API or Ollama.
- **Dynamic Content Generation**: Uses AI to generate contextually relevant example questions and UI content based on your PDF.

---

## Example Usage

1. Launch the app with `streamlit run streamlit_app.py`.
2. Ask questions like "What is the EU AI Act?" or "What are high-risk AI systems?"
3. The assistant responds, providing references to specific parts of the document.

**Using a Custom Document:**
1. Place your PDF in the project directory (e.g., `CompanyPolicy.pdf`)
2. Set `PDF_FILENAME=CompanyPolicy.pdf` in your `.env` file
3. Run the app - it will automatically adapt all UI elements to your document
4. The AI will generate relevant example questions based on your document's content

---

## Notes

- Ensure you have your PDF document in the project directory for document embedding (default: `EU_AI_Act.pdf`).
- The app will automatically detect which backend to use based on your `.env` configuration.
- **OpenAI Mode**: Uses `gpt-4o-mini` model and OpenAI embeddings for optimal performance.
- **Ollama Mode**: Uses local models (default: `llama3.2`) and local sentence-transformer embeddings, providing privacy and no API costs.
- **Custom PDF**: The UI dynamically adapts to any PDF you configure, including AI-generated example questions.
- You can switch between OpenAI and Ollama, or change the PDF document, by updating your `.env` file and restarting the app.

---

## License
This project is licensed under the MIT License.
