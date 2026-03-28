import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from ui.chat import (
    init_chat_history,
    display_chat_history,
    add_message,
    render_chat_input,
    display_sources
)
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
api_key = os.getenv("api key")





st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("RAG-based Chatbot")
st.markdown("Upload any PDF and ask any questions regarding the information on the PDF")

# initializing chat history on every run
init_chat_history()

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

            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=api_key,
                temperature=0,
                convert_system_message_to_human=True
            )

            template = """Use only the following context to answer the question and add few little extra words and sentences so the answer feels broader. No hallucination. 
            You can add bullet points if needed but if only needed. Make your answer comprehensible and use information from the context only. dont add anything that is not necessary just to make it broader.
            If you don't know or find any information, say "Did not find any information regarding your query."
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


if st.session_state.get("processed"):
    display_chat_history()
    question = render_chat_input()

    if question:
        add_message("user", question)
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.qa_chain.invoke(question)
                sources = st.session_state.retriever.invoke(question)
            st.markdown(answer)
            display_sources(sources)
         
        add_message("assistant", answer)

else:
    st.info("⬅️ Upload a PDF from the sidebar to start chatting.")
