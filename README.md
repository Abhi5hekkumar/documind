# 📄 DocuMind — AI-Powered PDF Chat Assistant

> A Retrieval-Augmented Generation (RAG) application that enables users to upload PDF documents and interact with them using natural language. Built with **LangChain**, **ChromaDB**, **Mistral AI**, and **Streamlit**.

---

## 🚀 Overview

DocuMind leverages Retrieval-Augmented Generation (RAG) to provide context-aware answers directly from uploaded PDF documents. Instead of relying on the language model's internal knowledge, the application retrieves the most relevant document chunks using semantic search and generates accurate responses grounded in the document.

This project demonstrates a complete end-to-end RAG pipeline including document ingestion, chunking, vector embeddings, semantic retrieval, and LLM-powered response generation.

---

## ✨ Features

- 📤 Upload any PDF document
- 💬 Ask questions in natural language
- 🧠 Retrieval-Augmented Generation (RAG)
- 🔍 Semantic search using Mistral Embeddings
- 📚 Automatic document chunking
- ⚡ Max Marginal Relevance (MMR) retrieval
- 💾 Chroma Vector Database
- 🤖 Mistral Large Language Model
- 🌐 Clean Streamlit Web Interface
- 🔄 Dynamic vector database creation on upload

---

# 🏗️ Architecture

```text
                 ┌────────────────────┐
                 │     Upload PDF     │
                 └──────────┬─────────┘
                            │
                            ▼
                 ┌────────────────────┐
                 │    PDF Loader      │
                 │    (PyPDFLoader)   │
                 └──────────┬─────────┘
                            │
                            ▼
                 ┌────────────────────┐
                 │ Document Chunking  │
                 │ Recursive Splitter │
                 └──────────┬─────────┘
                            │
                            ▼
                 ┌────────────────────┐
                 │ Mistral Embeddings │
                 └──────────┬─────────┘
                            │
                            ▼
                 ┌────────────────────┐
                 │     ChromaDB       │
                 │   Vector Store     │
                 └──────────┬─────────┘
                            │
                            ▼
                 ┌────────────────────┐
                 │   MMR Retriever    │
                 └──────────┬─────────┘
                            │
                            ▼
                 ┌────────────────────┐
                 │   Prompt Template  │
                 └──────────┬─────────┘
                            │
                            ▼
                 ┌────────────────────┐
                 │  Mistral Chat LLM  │
                 └──────────┬─────────┘
                            │
                            ▼
                 ┌────────────────────┐
                 │   AI Response      │
                 └────────────────────┘
```

---

# 🛠️ Tech Stack

| Category | Technology |
|-----------|------------|
| Language | Python |
| LLM | Mistral AI |
| Framework | LangChain |
| Vector Database | ChromaDB |
| Embedding Model | Mistral Embeddings |
| UI | Streamlit |
| PDF Loader | PyPDFLoader |
| Chunking | RecursiveCharacterTextSplitter |

---

# 📂 Project Structure

```text
DocuMind/
│
├── app.py                     # Streamlit Application
├── create_database.py         # Build Vector Database
├── document_loaders/
│   └── sample.pdf
│
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── images/
    └── demo.png
```

---

# ⚙️ Installation

## Clone the Repository

```bash
git clone https://github.com/<your-username>/DocuMind.git
cd DocuMind
```

## Create a Virtual Environment

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### macOS / Linux

```bash
source .venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configure Environment Variables

Create a `.env` file.

```env
MISTRAL_API_KEY=your_api_key_here
```

---

## Run the Application

```bash
streamlit run app.py
```

Open your browser and visit

```
http://localhost:8501
```

---

# 🔍 How It Works

### 1. Upload PDF

The user uploads a PDF through the Streamlit interface.

↓

### 2. Document Processing

The PDF is parsed using **PyPDFLoader**.

↓

### 3. Chunking

The document is divided into overlapping chunks using **RecursiveCharacterTextSplitter**.

↓

### 4. Embedding Generation

Each chunk is converted into dense vector embeddings using **Mistral Embeddings**.

↓

### 5. Vector Storage

Embeddings are stored inside **ChromaDB**.

↓

### 6. Semantic Retrieval

When the user asks a question, the retriever performs **MMR similarity search** to find the most relevant chunks.

↓

### 7. Response Generation

Retrieved context and the user's query are sent to **Mistral AI**, which generates the final grounded response.

---

# 📸 Demo

> Add screenshots of your application here.

```
images/
    demo.png
```

Example:

![Demo](images/demo.png)

---

# 🚀 Future Improvements

- Multiple PDF support
- Persistent Vector Database
- Source citations with page numbers
- Streaming responses
- Conversation memory
- Hybrid Search (BM25 + Vector Search)
- Cross-document querying
- Authentication
- Docker deployment
- Cloud deployment (AWS / Azure / GCP)

---

# 📊 Learning Outcomes

This project demonstrates practical experience with:

- Retrieval-Augmented Generation (RAG)
- Semantic Search
- Vector Databases
- Prompt Engineering
- Large Language Models
- Embedding Models
- LangChain Pipelines
- Streamlit Application Development
- Information Retrieval

---

# 🤝 Contributing

Contributions, feature requests, and suggestions are welcome.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a Pull Request

---

# 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Abhishek Kumar**

If you found this project useful, consider giving it a ⭐ on GitHub!
