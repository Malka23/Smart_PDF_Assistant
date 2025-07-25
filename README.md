# Smart_PDF_Assistant
# 📄 Smart PDF Q&A Assistant

The Smart PDF Assistant is an AI-powered tool that allows users to upload PDF documents and ask questions about their content. Built using Retrieval-Augmented Generation (RAG) and LLMs, this assistant extracts, indexes, and answers user queries from PDFs with high accuracy and context awareness.

---

## 🚀 Features

- 📄 Upload any PDF file (up to 200MB)
- 🔍 Automatically extracts and chunks text
- 🧠 Embeds chunks into a vector store for semantic search
- 💬 Ask natural language questions about the PDF content
- 🧠 Powered by Hugging Face LLMs (e.g., Flan-T5, Mistral)
- 🖥️ Deployed with Streamlit

---

## 🧰 Tech Stack
- Frontend: Streamlit

- Backend: Python

- Core Libraries:

- PyPDF2 or pdfplumber – PDF text extraction

- langchain – RAG pipeline (chunking, embeddings, retrieval)

- faiss – Vector similarity search

- transformers / OpenAI – LLMs for answer generation

- streamlit – Web UI

## 🧠 How It Works
Extracts text from the uploaded PDF.

Chunks the text for better embedding.

Embeds the chunks into a vector store using sentence-transformers.

Retrieves relevant chunks based on the user question.

Generates an answer using a Hugging Face LLM.

## 🤝 Contributing
We welcome contributions and suggestions! Please fork the repo, make your changes, and submit a pull request.

## 🚀 How to Run Locally
1. Clone the repository
2. Create a virtual environment
3. Install dependencies
4. Set up your .env file
5. Run the app
