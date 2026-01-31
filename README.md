# 🏥 Medical Chatbot with LLMs, LangChain & Pinecone

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.1.1-green.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1.0+-orange.svg)
![Pinecone](https://img.shields.io/badge/Pinecone-Vector%20DB-purple.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**An intelligent medical information assistant powered by Retrieval Augmented Generation (RAG) that provides accurate, evidence-based responses from medical documents.**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture) • [Contributing](#-contributing)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [API Endpoints](#-api-endpoints)
- [How It Works](#-how-it-works)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)

---

## 🎯 Overview

This Medical Chatbot is a sophisticated RAG (Retrieval Augmented Generation) application that enables users to query medical information from PDF documents. It leverages:

- **LangChain** for orchestrating LLM workflows
- **Pinecone** as a vector database for semantic search
- **Groq** for fast LLM inference
- **HuggingFace Embeddings** for document vectorization
- **Flask** for the web interface

The chatbot provides evidence-based medical information by retrieving relevant context from medical documents and generating accurate responses using advanced language models.

---

## ✨ Features

- 🔍 **Semantic Search**: Uses vector embeddings to find the most relevant medical information
- 📚 **PDF Document Processing**: Automatically extracts and processes medical PDFs
- 🤖 **Intelligent Responses**: Provides context-aware, evidence-based answers
- 💬 **Interactive Web Interface**: Beautiful, responsive chat UI
- ⚡ **Fast Inference**: Powered by Groq for quick response times
- 🔒 **Context-Aware**: Only uses information from provided medical documents
- 📊 **Chunking Strategy**: Optimized text splitting for better retrieval

---

## 🛠 Tech Stack

### Backend
- **Flask** - Web framework
- **LangChain** - LLM orchestration framework
  - `langchain-core` - Core LangChain functionality
  - `langchain-community` - Community integrations
  - `langchain-classic` - Classic chain patterns
  - `langchain-pinecone` - Pinecone vector store integration
  - `langchain-groq` - Groq LLM integration
  - `langchain-huggingface` - HuggingFace embeddings
  - `langchain-text-splitters` - Text splitting utilities

### Vector Database & Embeddings
- **Pinecone** - Managed vector database
- **HuggingFace Embeddings** - `sentence-transformers/all-MiniLM-L6-v2`

### LLM Provider
- **Groq** - Fast LLM inference API

### Frontend
- **HTML5/CSS3** - Modern web interface
- **Bootstrap 4** - Responsive design
- **jQuery** - DOM manipulation and AJAX

---

## 🏗 Architecture

```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Flask App     │
│   (app.py)      │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│   LangChain RAG Pipeline        │
│                                  │
│   1. Query Embedding            │
│   2. Vector Search (Pinecone)   │
│   3. Context Retrieval           │
│   4. LLM Generation (Groq)       │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────┐
│   Response      │
└─────────────────┘
```

### Data Flow

1. **Document Processing** (`store_index.py`):
   - Loads PDF documents from `data/` directory
   - Splits documents into chunks (500 chars, 20 overlap)
   - Generates embeddings using HuggingFace model
   - Stores vectors in Pinecone

2. **Query Processing** (`app.py`):
   - User submits query via web interface
   - Query is embedded using the same model
   - Semantic search retrieves relevant chunks from Pinecone
   - Retrieved context + query → LLM (Groq)
   - Generated response returned to user

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- Pinecone account and API key
- Groq API key
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/Prakharjain1211/Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask.git
cd Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables

Create a `.env` file in the root directory:

```env
PINECONE_API_KEY=your_pinecone_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

### Step 5: Prepare Medical Documents

Place your medical PDF files in the `data/` directory:

```bash
data/
  └── Medical_book.pdf
```

### Step 6: Create Vector Store

Run the indexing script to process PDFs and create the Pinecone index:

```bash
python store_index.py
```

This will:
- Load all PDFs from `data/` directory
- Split them into chunks
- Generate embeddings
- Create/update Pinecone index named `medical-chatbot`

---

## ⚙️ Configuration

### Pinecone Index Settings

The default configuration in `store_index.py`:

- **Index Name**: `medical-chatbot`
- **Dimension**: 384 (matches HuggingFace model)
- **Metric**: `cosine`
- **Cloud**: AWS
- **Region**: us-east-1

### Embedding Model

Default model: `sentence-transformers/all-MiniLM-L6-v2`
- Dimension: 384
- Fast and efficient for semantic search

### Text Splitting

- **Chunk Size**: 500 characters
- **Chunk Overlap**: 20 characters

You can modify these in `src/helper.py`:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # Adjust as needed
    chunk_overlap=20     # Adjust as needed
)
```

### System Prompt

Customize the medical assistant's behavior in `src/prompt.py`.

---

## 🚀 Usage

### Starting the Application

```bash
python app.py
```

The Flask app will start on `http://localhost:5000` (or the port specified in your Flask configuration).

### Accessing the Chat Interface

1. Open your web browser
2. Navigate to `http://localhost:5000`
3. Start chatting with the medical assistant!

### Example Queries

- "What are the symptoms of diabetes?"
- "Explain the mechanism of action of aspirin"
- "What are the contraindications for this medication?"
- "Describe the diagnostic criteria for hypertension"

---

## 📁 Project Structure

```
Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask/
│
├── app.py                 # Main Flask application
├── store_index.py         # Script to process PDFs and create vector store
├── setup.py               # Package setup configuration
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (create this)
├── README.md              # This file
│
├── data/                  # Medical PDF documents
│   └── Medical_book.pdf
│
├── src/                   # Source code modules
│   ├── __init__.py
│   ├── helper.py         # PDF processing and embedding functions
│   └── prompt.py         # System prompt for medical assistant
│
├── frontend/              # Frontend templates
│   └── chat.html         # Chat interface
│
├── static/               # Static files (CSS, JS, images)
│   └── style.css         # Chat interface styling
│
└── research/             # Research and experimentation
    └── trials.ipynb      # Jupyter notebook for testing
```

---

## 🔌 API Endpoints

### `GET /`
- **Description**: Renders the chat interface
- **Response**: HTML page with chat UI

### `POST /get`
- **Description**: Processes user query and returns AI response
- **Request Body**:
  ```json
  {
    "msg": "What are the symptoms of diabetes?"
  }
  ```
- **Response**:
  ```json
  "Based on the medical context provided, diabetes symptoms include..."
  ```

---

## 🔬 How It Works

### 1. Document Indexing (`store_index.py`)

```python
# Load PDFs
extracted_data = loadPDFFile(data="data/")

# Filter and clean documents
filter_data = filterToMinimalDocs(extracted_data)

# Split into chunks
text_chunks = textSplit(filter_data)

# Generate embeddings and store in Pinecone
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name="medical-chatbot",
    embedding=embeddings
)
```

### 2. Query Processing (`app.py`)

```python
# Create retrieval chain
retrieval_chain = create_retrieval_chain(
    retriever=vector_store.as_retriever(),
    llm_chain=document_chain
)

# Process query
response = retrieval_chain.invoke({
    "input": user_query
})
```

### 3. RAG Pipeline

1. **Retrieval**: Query is embedded and searched in Pinecone
2. **Augmentation**: Retrieved chunks are combined with the query
3. **Generation**: LLM generates response using retrieved context

---

## 🧪 Development

### Running Tests

```bash
# Activate virtual environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Run the application
python app.py
```

### Modifying the System Prompt

Edit `src/prompt.py` to change the assistant's behavior:

```python
system_prompt = (
    "Your custom prompt here..."
)
```

### Adding New Documents

1. Add PDF files to `data/` directory
2. Run `python store_index.py` to re-index

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Guidelines

- Follow PEP 8 style guidelines
- Add comments for complex logic
- Update README if adding new features
- Test your changes thoroughly

---

## ⚠️ Important Notes

### Medical Disclaimer

**This chatbot is for informational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified health providers with any questions you may have regarding a medical condition.**

### Limitations

- Responses are based solely on provided documents
- May not have information on all medical topics
- Should not be used for emergency medical situations
- Accuracy depends on the quality of source documents

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Prakhar Jain**

- Email: prakharjain1211@gmail.com
- GitHub: [@Prakharjain1211](https://github.com/Prakharjain1211)

---

## 🙏 Acknowledgments

- [LangChain](https://www.langchain.com/) for the amazing LLM orchestration framework
- [Pinecone](https://www.pinecone.io/) for the vector database infrastructure
- [Groq](https://groq.com/) for fast LLM inference
- [HuggingFace](https://huggingface.co/) for the embedding models
- [Flask](https://flask.palletsprojects.com/) for the web framework

---

<div align="center">

**⭐ If you find this project helpful, please consider giving it a star! ⭐**

Made with ❤️ for the medical community

</div>
