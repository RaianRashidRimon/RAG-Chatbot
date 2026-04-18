from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage


def build_qa_chain(retriever, api_key: str, model_name: str = "gemini-2.5-flash-lite"):
    """Build a conversational QA chain with manual memory handling.

    Args:
        retriever: the vector store retriever
        api_key: Gemini API key
        model_name: Gemini model to use

    Returns:
        A callable chain that accepts {"question": str, "chat_history": list}
    """
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0,
        convert_system_message_to_human=True
    )
    condense_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the conversation history and a follow-up question, "
                   "rephrase the follow-up into a standalone question. "
                   "If it's already standalone, return it as-is."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    condense_chain = condense_prompt | llm | StrOutputParser()



    
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant. Use only the following context to answer.
Add a few connecting words so the answer feels natural, but do not hallucinate.
Use bullet points only if genuinely needed.
If you cannot find the answer say: "Did not find any information regarding your query."

Context:
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def run_chain(inputs: dict) -> dict:
        question = inputs["question"]
        chat_history = inputs.get("chat_history", [])
        if chat_history:
            standalone_question = condense_chain.invoke({
                "question": question,
                "chat_history": chat_history
            })
        else:
            standalone_question = question
        docs = retriever.invoke(standalone_question)
        context = format_docs(docs)
        answer = (answer_prompt | llm | StrOutputParser()).invoke({
            "question": question,
            "chat_history": chat_history,
            "context": context
        })
        return {
            "answer": answer,
            "source_documents": docs
        }
    return run_chain
