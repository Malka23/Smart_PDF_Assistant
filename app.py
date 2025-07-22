from rag_utils import ask_question
import streamlit as st
from rag_utils import extract_text_from_pdf, chunk_text, embed_chunks, create_faiss_index

st.set_page_config(page_title="📄 Smart PDF Q&A Assistant")
st.title("📄 Smart PDF Q&A Assistant")
st.markdown("Upload a PDF and start asking questions (RAG-powered).")

pdf_file = st.file_uploader("Upload a PDF", type="pdf")

if pdf_file:
    with st.spinner("Extracting text..."):
        full_text = extract_text_from_pdf(pdf_file)
        st.success("✅ Text extracted!")

    with st.expander("📖 View Extracted Text"):
        st.write(full_text[:2000] + "..." if len(full_text) > 2000 else full_text)

    with st.spinner("Chunking and embedding..."):
        chunks = chunk_text(full_text)
        vectors = embed_chunks(chunks)
        index = create_faiss_index(vectors)
        st.success(f"✅ {len(chunks)} chunks embedded and indexed!")

    st.success("Now ready for Q&A! (Next step)")

    # Step 4: Ask Questions
    st.header("Ask Questions")
    question = st.text_input("Enter your question about the PDF:")
    from langchain.vectorstores import FAISS  # or Chroma, etc.
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.document_loaders import TextLoader
    from langchain.text_splitter import CharacterTextSplitter

    # Load and split documents
    loader = TextLoader("your_pdf_or_text_file.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # Create or load vectorstore
    embedding = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embedding)  # or load from disk if already saved

    if question:
        response = ask_question(vectorstore, question)
        st.markdown("**Answer:**")
        st.write(response)



