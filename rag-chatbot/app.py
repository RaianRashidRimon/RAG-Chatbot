import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from core.chain import build_qa_chain
import tempfile
import os


load_dotenv()
## create a '.env' file on the project folder and write " api key = {your actual api key} "
api_key = os.getenv("api key") ## -> paste your api key variable from .env file






st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("RAG-based Chatbot")
st.markdown("Upload any PDF and ask any questions regarding the information on the PDF")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.header("Upload document")
    uploaded_file = st.file_uploader("Choose your file", type="pdf")

    if st.button("Upload") and uploaded_file:
        with st.spinner("Processing your file...."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            text_splitters = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = text_splitters.split_documents(docs)

            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            db = Chroma.from_documents(chunks, embeddings)
            retriever = db.as_retriever(search_kwargs={"k": 3})

            qa_chain = build_qa_chain(retriever, api_key)

            st.session_state.qa_chain = qa_chain
            st.session_state.retriever = retriever
            st.session_state.processed = True 
            st.success("PDF processed! Ask questions below.")
        os.unlink(tmp_path)


if st.session_state.get("processed"):
    
    # render previous messages from history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📄 Sources (click to view)"):
                    for i, doc in enumerate(message["sources"]):
                        page = doc.metadata.get('page', 0) + 1
                        st.markdown(f"**Source {i+1} — Page {page}**")
                        st.code(doc.page_content.strip()[:500] + "...", language="text")
                        if i < len(message["sources"]) - 1:
                            st.divider()
    question = st.chat_input("Ask anything about your document...")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.qa_chain.invoke({"question": question})
                answer = result["answer"]
                sources = result["source_documents"]
            st.markdown(answer)
            with st.expander("📄 Sources (click to view)"):
                for i, doc in enumerate(sources):
                    page = doc.metadata.get('page', 0) + 1
                    st.markdown(f"**Source {i+1} — Page {page}**")
                    st.code(doc.page_content.strip()[:500] + "...", language="text")
                    if i < len(sources) - 1:
                        st.divider()
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })

else:
    st.info("Upload a PDF from the sidebar to get started")
