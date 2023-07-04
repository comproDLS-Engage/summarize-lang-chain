from langchain import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

llm = OpenAI(temperature=0)

def summarize(text_file):
    if type(text_file) == str:
        with open(text_file, 'r') as file:
            essay = file.read()
    else:
        essay = text_file.read()

    summary_template = """
    Please write a one sentence summary of the following text:

    {essay}
    """

    summary_prompt = PromptTemplate(
        input_variables=["essay"],
        template=summary_template
    )

    summary_chain = LLMChain(prompt=summary_prompt, llm=llm, output_key="summary")

    response = summary_chain.run(essay)

    return response

if __name__ == "__main__":
    text_file = 'temp.txt'
    print(summarize(text_file))

