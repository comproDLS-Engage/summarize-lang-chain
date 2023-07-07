from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from langchain.utilities import GoogleSearchAPIWrapper

import os
os.environ['OPENAI_API_KEY'] = "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"
os.environ['SERPAPI_API_KEY'] = "5000e62d0511132a87889732f68f8cf884636841636d36f1327359a47c044a5d"



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


												
