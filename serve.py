from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langserve import add_routes
import os
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq LLM
model = ChatGroq(model="openai/gpt-oss-20b", groq_api_key=groq_api_key)

# Initialize String output parser
parser = StrOutputParser()

# Define PromptTemplate
generic_template = "Translate the following into {language}:"

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", generic_template),
        ("user", "{text}")
    ]
)

# Define Chain
prompt_chain = prompt|model|parser

app = FastAPI(title="Langchain Server", version="1.0", description="Simple api server using langchain runnable interfaces")

# App Definition

add_routes(
    app,
    prompt_chain,
    path = "/prompt_chain"
)

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
