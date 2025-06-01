from langchain_google_genai import ChatGoogleGenerativeAI
import os
llm=ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=os.environ["GOOGLE_API_KEY"]
    
)

__all__=["llm"]