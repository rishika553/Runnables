from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain.tools import tool
import os
from rich import print
from langchain_core.messages import HumanMessage

#1 creating a tool
@tool
def get_text_length(text:str) -> int:
    """Returns the number of character in a given text """
    return len(text)
tools = {
  "get_text_length": get_text_length
}
llm = ChatOpenAI(
    model="openrouter/free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)
#tool binding 
llm_with_tool = llm.bind_tools([get_text_length])
message  = []
query = HumanMessage("Return the number of characters in the given text :'hello how are you'")
message.append(query)

result = llm_with_tool.invoke(message)

message.append(result)

if result.tool_calls:
  
    tool_name = result.tool_calls[0]["name"]
    tool_message = tools[tool_name].invoke(result.tool_calls[0])
    message.append(tool_message)
    print(message)