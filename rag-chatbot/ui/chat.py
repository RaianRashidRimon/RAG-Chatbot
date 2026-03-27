import streamlit as st




def init_chat_history():
    """Initialize chat history in session state if it doesn't exist."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def display_chat_history():
    """Render all previous messages from chat history."""
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def add_message(role: str, content: str):
    """Add a new message to the chat history.
    
    Args:
        role: either "user" or "assistant"
        content: the message text
    """
    st.session_state.chat_history.append({
        "role": role,
        "content": content
    })

def render_chat_input():
    """Render the chat input box and handle submission.
    
    Returns:
        str | None: the user's question if submitted, otherwise None
    """
    return st.chat_input("Ask anything about your document...")

def display_sources(sources: list):
    """Render the source chunks used to generate the answer.
    
    Args:
        sources: list of LangChain Document objects
    """
    with st.expander("📄 Sources (click to view)"):
        for i, doc in enumerate(sources):
            page = doc.metadata.get('page', 0) + 1
            st.markdown(f"**Source {i+1} — Page {page}**")
            st.code(doc.page_content.strip()[:500] + "...", language="text")
            if i < len(sources) - 1:
                st.divider()
