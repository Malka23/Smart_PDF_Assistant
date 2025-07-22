import os
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# rag_utils.py
from langchain_community.llms import HuggingFaceHub

llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",
    model_kwargs={"temperature": 0.7, "max_length": 500},
    task="text2text-generation"
)
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub

load_dotenv()

def extract_text_from_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

def chunk_text(documents):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

def embed_chunks(chunks):
    embeddings = HuggingFaceEmbeddings()
    return FAISS.from_documents(chunks, embeddings)

def ask_question(vectorstore, question):
    repo_id = "tiiuae/falcon-7b-instruct"
    llm = HuggingFaceHub(
        repo_id=repo_id,
        model_kwargs={"temperature": 0.5, "max_new_tokens": 512}
    )
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
    return qa.run(question)
