from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from langchain.utilities import GoogleSearchAPIWrapper

import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())



llm = OpenAI(temperature=0)

tool_names = ["serpapi"]
tools = load_tools(tool_names)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description")

def get_answers(query):
    response = agent.run(query)
    return response



if __name__ == "__main__":
    query = "Where is Eiffel Tower located?"
    print(get_answers(query))


												
