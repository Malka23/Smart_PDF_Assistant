import streamlit as st
import os
from rag_utils import extract_text_from_pdf, chunk_text, embed_chunks, ask_question

st.set_page_config(page_title="Smart PDF Q&A Assistant")
st.title("📄 Smart PDF Q&A Assistant")
st.markdown("Upload a PDF and start asking questions (RAG-powered).")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    docs = extract_text_from_pdf("temp.pdf")
    st.success("✅ Text extracted!")

    chunks = chunk_text(docs)
    vectorstore = embed_chunks(chunks)
    st.success("✅ Chunks embedded and indexed!")
    st.session_state.vectorstore = vectorstore

    st.success("Now ready for Q&A! (Next step)")

# Ask Questions UI
if "vectorstore" in st.session_state:
    st.header("❓ Ask Questions")
    question = st.text_input("Ask your question about the PDF:")

    if question:
        response = ask_question(st.session_state.vectorstore, question)
        st.markdown(f"**🤖 Answer:** {response}")

