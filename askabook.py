# PDF Loaders.
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

loader = PyPDFLoader("./data/field-guide-to-data-science.pdf")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
texts = text_splitter.split_documents(data)


# Check to see if there is an environment variable with you API keys, if not, use what you put below
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
PINECONE_API_KEY = "0e71ade9-45c7-42b6-882b-5cfeea7b0a4f"
PINECONE_API_ENV = "us-west4-gcp-free"

embeddings = OpenAIEmbeddings()

# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
)
index_name = "langchaintest1"  # put in the name of your pinecone index here


docsearch = Pinecone.from_existing_index(index_name, embeddings)


def get_answers(query):
    # query = "What are examples of good data science teams?"
    docs = docsearch.similarity_search(query)
    llm = OpenAI(temperature=0)
    chain = load_qa_chain(llm, chain_type="stuff")
    result = chain.run(input_documents=docs, question=query)

    return result

if __name__ == "__main__":
    query = "What are examples of good data science teams?"
    print(get_answers(query))



