from dotenv import load_dotenv
import os
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough


#components

model = ChatOpenAI(
    model="openrouter/free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)
parser= StrOutputParser()

code_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a code generator"),
    ("human","{topic}")
])

explain_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant who explains code in simple terms "),
    ("human","explain the following code in simple words:\n{code}")
])
seq = code_prompt | model | parser 

seq2 = RunnableParallel(
    {
        "code": RunnablePassthrough(),
        "explaination": explain_prompt | model | parser
    }
)

chain = seq | seq2

result = chain.invoke({"topic": "please write a code of palindrome in python  "})

print(result['code'])
print(result['explaination'])
