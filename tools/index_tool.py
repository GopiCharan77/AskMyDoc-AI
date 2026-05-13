from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from models.EM import embedding

def indexer(file):
    docs = PyMuPDFLoader(file).load()
    splits = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1024, chunk_overlap=128
    ).split_documents(docs)
    
    batch_size = 16
    vectorstore = FAISS.from_documents(splits[:batch_size], embedding)
    
    for i in range(batch_size, len(splits), batch_size):
        vectorstore.add_documents(splits[i:i+batch_size])
        
    vectorstore.save_local("index")
