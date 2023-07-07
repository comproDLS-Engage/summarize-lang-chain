from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.evaluation.qa import QAEvalChain

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


prompt = PromptTemplate(
    template="Question: {question}\nAnswer:", input_variables=["question"]
)


llm = OpenAI(temperature=0)
chain = LLMChain(llm=llm, prompt=prompt)


def evaluate(examples):
    predictions = chain.apply(examples)
    eval_chain = QAEvalChain.from_llm(llm)
    graded_outputs = eval_chain.evaluate(
        examples, predictions, question_key="question", prediction_key="text"
    )
    return {"examples": examples, "predictions": predictions, "graded_outputs": graded_outputs}


if __name__ == '__main__':
    _examples = [
        {
            "question": "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?",
            "answer": "11",
        },
        {
            "question": 'Is the following sentence plausible? "Joao Moutinho caught the screen pass in the NFC championship."',
            "answer": "No",
        },
    ]
    response = evaluate(_examples)
    examples = response["examples"]
    predictions = response["predictions"]
    graded_outputs = response["graded_outputs"]
    for i, eg in enumerate(examples):
        print(f"Example {i + 1}:")
        print("Question: " + eg["question"])
        print("Real Answer: " + eg["answer"])
        print("Predicted Answer: " + predictions[i]["text"])
        print("Predicted Grade: " + graded_outputs[i]["text"])
        print()
