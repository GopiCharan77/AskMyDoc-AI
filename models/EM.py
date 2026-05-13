from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
import os

embedding = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2", google_api_key=os.environ.get("GEMINI_API_KEY")
)
