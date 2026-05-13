from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
import os

llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview", 
    google_api_key=os.environ.get("GEMINI_API_KEY"),
    thinking_level="low",
)
