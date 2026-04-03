from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate


def build_qa_chain(retriever, api_key: str, model_name: str = "gemini-2.5-flash-lite"):
    """Build a conversational QA chain with memory.

    Args:
        retriever: the vector store retriever
        api_key: Gemini API key
        model_name: Gemini model to use

    Returns:
        ConversationalRetrievalChain with memory attached
    """
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0,
        convert_system_message_to_human=True
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    answer_prompt = PromptTemplate.from_template("""
              You are a helpful assistant. Use only the following context to answer the question.
              Add a few connecting words so the answer feels natural, but do not hallucinate or
              add information not found in the context. Use bullet points only if genuinely needed.
              If you cannot find the answer, say "Did not find any information regarding your query."

      Context:
      {context}

      Question: {question}
      Answer:""")

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": answer_prompt}
    )
    return qa_chain
