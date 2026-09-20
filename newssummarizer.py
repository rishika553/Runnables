from dotenv import load_dotenv
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI

load_dotenv()

# 1. Initialize Tavily Search
search_tool = TavilySearch(max_results=5)

# 2. Initialize OpenRouter LLM
llm = ChatOpenAI(
    model="openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0
)

# 3. Create the prompt
prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful AI news summarizer.

    Summarize the following news into clear, concise bullet points.

    Include only facts supported by the search results.
    Highlight the most important developments.
    Do not invent information.

    News:
    {news}
    """
)

# 4. Create the LCEL chain
chain = prompt | llm | StrOutputParser()

# 5. Search for news
print("Searching for the latest AI news...")

news_result = search_tool.invoke(
    {"query": "latest AI news 2026"}
)

print("\nSearch completed!")

# 6. Generate the summary
print("\nGenerating summary...")

result = chain.invoke({"news": news_result})

# 7. Display the summary
print("\n===== AI NEWS SUMMARY =====\n")
print(result)
