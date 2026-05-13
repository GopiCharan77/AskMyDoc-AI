from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
import os

class SafeBatchGoogleEmbeddings(GoogleGenerativeAIEmbeddings):
    def embed_documents(self, texts: list[str], *args, **kwargs) -> list[list[float]]:
        # Override to strictly enforce the API's max batch size limit
        # This resolves silent truncation bugs in FAISS
        return super().embed_documents(texts, batch_size=100)

embedding = SafeBatchGoogleEmbeddings(
    model="models/gemini-embedding-2", google_api_key=os.environ.get("GEMINI_API_KEY")
)
