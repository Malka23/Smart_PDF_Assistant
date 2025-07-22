# utils/rag_utils.py

import fitz  # PyMuPDF
import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def embed_chunks(chunks):
    vectors = embedder.encode(chunks)
    return np.array(vectors)

def create_faiss_index(vectors):
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index


from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFaceHub


def ask_question(vectorstore, question):
    # Replace with your actual model name on Hugging Face Hub
    llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature": 0.5, "max_length": 512})

    docs = vectorstore.similarity_search(question)
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain.run(input_documents=docs, question=question)
