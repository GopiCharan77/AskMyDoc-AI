# AskMyDoc AI

An Agentic RAG (Retrieval-Augmented Generation) app with self-reflection and corrective capabilities.

## Features
- Upload any PDF and ask questions about it
- Automatically grades retrieved chunks for relevance
- Falls back to live web search if document is not relevant
- Powered by Groq (Llama 3.1) + FAISS + LangChain

## Tech Stack
- LLM: Llama 3.1 8B via Groq
- Embeddings: HuggingFace all-MiniLM-L6-v2
- Vector Store: FAISS
- Web Search: Tavily / DuckDuckGo
- Framework: LangChain + LangGraph
- UI: Streamlit
- Deployment: Hugging Face Spaces (Docker)

## Live Demo
https://huggingface.co/spaces/gopicharan7/Self_Reflective_RAG
