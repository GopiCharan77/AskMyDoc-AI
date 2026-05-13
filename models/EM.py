from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
import os

embedding = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004", google_api_key=os.environ.get("GEMINI_API_KEY")
)
