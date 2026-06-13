import streamlit as st
from dotenv import load_dotenv
from graph import workflow_compiler
rag_graph = workflow_compiler()
load_dotenv()

st.set_page_config(page_title="Agentic CRAG", page_icon="🔍")
st.title("🔍 Agentic RAG with Self-Reflection")
st.caption("Corrective RAG — retrieves, grades docs, and web-searches if needed")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.chat_input("Ask me anything...")

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if query:
    st.session_state.history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = rag_graph.invoke({"question": query})
            answer = result.get("generation", "No answer generated.")
        st.write(answer)
    st.session_state.history.append({"role": "assistant", "content": answer})
