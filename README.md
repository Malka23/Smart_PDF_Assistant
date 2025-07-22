# Smart_PDF_Assistant
# 📄 Smart PDF Q&A Assistant

A GenAI-powered application that allows users to upload a PDF and ask questions based on its content using **RAG (Retrieval-Augmented Generation)** with **Hugging Face models**.

---

## 🚀 Features

- 📄 Upload any PDF file (up to 200MB)
- 🔍 Automatically extracts and chunks text
- 🧠 Embeds chunks into a vector store for semantic search
- 💬 Ask natural language questions about the PDF content
- 🧠 Powered by Hugging Face LLMs (e.g., Flan-T5, Mistral)
- 🖥️ Deployed with Streamlit

---

## 📦 Folder Structure

Smart_PDF_Assistant/
│
├── app.py # Streamlit app
├── rag_utils.py # PDF processing, embedding & QA logic
├── requirements.txt # Python dependencies

## 🧠 How It Works
Extracts text from the uploaded PDF.

Chunks the text for better embedding.

Embeds the chunks into a vector store using sentence-transformers.

Retrieves relevant chunks based on the user question.

Generates an answer using a Hugging Face LLM.

## 📚 Models & Tools Used
🔗 Hugging Face Transformers

🧠 LangChain

🧩 Sentence Transformers

📄 PyMuPDF / fitz

🖥️ Streamlit
