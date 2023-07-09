from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
import nltk

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

nltk.download('averaged_perceptron_tagger')

# Get your loader ready
loader = DirectoryLoader('./data/cricket', glob='**/*.txt')

# Load up your text into documents
documents = loader.load()

# Get your text splitter ready
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=0)

# Split your documents into texts
texts = text_splitter.split_documents(documents)

# Turn your texts into embeddings
embeddings = OpenAIEmbeddings()

# Get your docsearch ready
docsearch = FAISS.from_documents(texts, embeddings)

# Load up your LLM
llm = OpenAI(temperature=0)

# Create your Retriever
qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())

qa = RetrievalQA.from_chain_type(llm=llm,
                                 chain_type="stuff",
                                 retriever=docsearch.as_retriever(),
                                 return_source_documents=True)

def get_answers(query):
    # query = "What is compro technologies?"
    response = qa({"query": query})

    return response

if __name__ == "__main__":
    query = "What is compro technologies?"
    print(get_answers(query))
