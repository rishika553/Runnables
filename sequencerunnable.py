from dotenv import load_dotenv
import os
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
 

# 1 prompt template 
prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple words"
)

# 2 model
model = ChatOpenAI(
    model="openrouter/free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)
# 3 output parser
parser = StrOutputParser()


chain = prompt |model |parser

result = chain.invoke("Genrative ai jobs in india 2026") 
print(result)
