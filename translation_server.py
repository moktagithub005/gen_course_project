# Important imports
import os 
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from fastapi import FastAPI
from langserve import add_routes
import uvicorn

# Load environment variables from .env file
load_dotenv()

# LangSmith tracking configuration
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "Translation-API"

# LLM setup using Groq
llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=1,
    api_key=os.getenv("GROQ_API_KEY")
)

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional translator. Translate the following text to {language}:"),
    ("human", "{text}")
]) 

# Output parser
parser = StrOutputParser()

# LangChain chain
chain = prompt | llm | parser

# Optional: Test function
def test_translation():
    result = chain.invoke({
        "language": "hindi",
        "text": "hello how are you today"
    })
    print(f"Result translation: {result}")

# Create FastAPI app
app = FastAPI(
    title="Translation API",
    version="1.0.0",
    description="Simple translation service with LangSmith tracking"
)

# Add LangServe endpoints
add_routes(
    app,
    chain,
    path="/translate",
    enable_feedback_endpoint=False,
    enable_public_trace_link_endpoint=False,
    playground_type="default"
)

# Health check endpoint
@app.get("/")
async def health_check():
    return {"status": "Translation API is running", "endpoint": "/translate"}

# Main execution block
if __name__ == "__main__":
    test_translation()
    print("API will be available at: http://127.0.0.1:8000")
    print("Translation endpoint: http://127.0.0.1:8000/translate")
    uvicorn.run(app, host="127.0.0.1", port=8001)

