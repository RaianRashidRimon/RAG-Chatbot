import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import tempfile
import os


load_dotenv()
## create a '.env' file on the project folder and write " api key = {your actual api key} "
api_key = os.getenv("api key") ## -> paste your api key variable from .env file


st.set_page_config(page_title="Chatbot", layout="wide")
st.title("RAG-based Chatbot")
st.markdown("Upload any PDF and ask any questions regarding the information on the PDF")

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
            
            ## embedding
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            ## vector database
            db = Chroma.from_documents(chunks, embeddings)
            retriever = db.as_retriever(search_kwargs={"k": 2})
            
            ## gemini llm
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash", ## -> you can add any gemini model here paid or free.
                google_api_key=api_key,
                temperature=0,
                convert_system_message_to_human = True
            )
            
            ## defining the prompt
            template = """Use only the following context to answer the question. Absolutely no hallucination. 
            If you don't know, say "Not found in policy."
            Context:
            {context}

            Question: {question}
            Answer:"""
            prompt = PromptTemplate.from_template(template)
            
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
            
            qa_chain = (
                {
                    "context": retriever | format_docs,
                    "question": RunnablePassthrough()
                }
                | prompt
                | llm
                | StrOutputParser()
            )
            
            st.session_state.qa_chain = qa_chain
            st.session_state.retriever = retriever
            st.session_state.processed = True
            st.success("PDF processed! Ask questions below.")
        os.unlink(tmp_path)

if st.session_state.get("Processed"):
    question = st.text_input("Ask anything:")
    if question:
        with st.spinner("Thinking...."):
            answer = st.session_state.qa_chain.invoke(question)
            sources = st.session_state.retriever.invoke(question)
        
        st.markdown(f"**Answer:** {answer}")
        
        with st.expander("Sources (Click to view)"):
            for i, doc in enumerate(sources):
                page = doc.metadata.get('page', 0) + 1
                st.markdown(f"**Source {i+1}: Page {page}**")
                st.code(doc.page_content.strip()[:500] + "...", language="text")
else:
    st.info("Upload a PDF to start.")
