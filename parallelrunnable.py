from dotenv import load_dotenv
import os
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableLambda
 #components
model = ChatOpenAI(
    model="openrouter/free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)
parser = StrOutputParser()

#Two different outputs 

short_prompt = ChatPromptTemplate.from_template(
    "explain {topic} in 1-2 lines"
)
detailed_prompt = ChatPromptTemplate.from_template(
    "Explain {topic in detail}"
)

#Input 
topic = "Generative AI " 


# formatted_short = short_prompt.format_messages(topic= topic)
# response_short = model.invoke(formatted_short)
# str_out = parser.parse(response_short.content)
#now we use parallel runnable

chain = RunnableParallel({
"short" : RunnableLambda(lambda x:x ['short']) |short_prompt | model|  parser,
"detailed" :RunnableLambda(lambda x:x ['detailed']) | detailed_prompt |model | parser
})

result = chain.invoke({
    "short ": {"topic": "genai"},
    "detailed":{"topic":"Deep learning"}
    })

print(result["short"])
print(result["detailes"])
