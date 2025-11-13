# RAG Chatbot
A **Retrieval-Augmented Generation (RAG)** chatbot that answers questions from PDFs with **citations**, without any hallucination.

## Features
1. Upload any PDF
2. Ask natural language questions
3. Get answers **only from the document**
4. See **source page and text**


## How To Run

1. Clone the repo

       git clone https://github.com/RaianRashidRimon/RAG-Chatbot.git    
and open the repo in Visual Studio

2. Install the dependecies

       pip install -r requirements.txt
   
4. Get your API key (paid or free) from Google/OpenAI/Meta(Llama)/Xai
5. Create .env file and paste

        api key = {your api key that youve got}
   
6. On the terminal run the app by writing

       streamlit run app.py

## Tech Stack
- Streamlit (UI)
- Google Gemini 2.5-Flash (LLM)
- HuggingFace al-MiniLM-L6-v2 (Embeddings)
- Chroma (Vector Database)
- LangChain (Backend Framework)
- PyPDFLoader (PDF Loader)
