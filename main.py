import streamlit as st
import summarize
import tempfile
import os
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import extraction
import Q_and_A_docs
import QAEvaluations
import SearchResults
import askabook


from tools import ImageCaptionTool, ObjectDetectionTool

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

document_type = st.sidebar.selectbox("Choose an Example", ("Summarize a Document", "Query Latest Data",
                                     "Query a Preloaded Book", "Q&A From Documents", "Extraction", "Q&A Evaluation"))

if document_type == "Summarize a Document":
    st.title("Upload a text file to generate it's summary.")

    file = st.file_uploader(label="Upload document",
                            accept_multiple_files=True, type=".txt")

    if file:
        response = summarize.summarize(file)
        for essay in response:
            st.header("Summary of " + essay["name"])
            st.write(essay["summary"])
elif document_type == "Describe an Image":
    ##############################
    ### initialize agent #########
    ##############################
    tools = [ImageCaptionTool(), ObjectDetectionTool()]

    conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=5,
        return_messages=True
    )

    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo"
    )

    agent = initialize_agent(
        agent="chat-conversational-react-description",
        tools=tools,
        llm=llm,
        max_iterations=5,
        verbose=True,
        memory=conversational_memory,
        early_stopping_method='generate'
    )

    # set title
    st.title('Generate description and identify objects in an image')

    # upload file
    file = st.file_uploader("Please upload an image",
                            type=["jpeg", "jpg", "png"])
    user_question_1 = "generate a caption for this image?"
    user_question_2 = "Please tell me what are the items present in the image."

    if file:
        # display image
        st.image(file, use_column_width=True)

        # Save the file to a temporary directory
        temp_dir = tempfile.TemporaryDirectory()
        file_path = os.path.join(temp_dir.name, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        # write agent response
        with st.spinner(text="In progress..."):
            response1 = agent.run(
                f'{user_question_1}, this is the image path: {file_path}')
            st.write("The description of the image is: " + response1)
            response2 = agent.run(
                f'{user_question_2}, this is the image path: {file_path}')
            st.write(response2)

            # Clean up the temporary directory
            temp_dir.cleanup()
elif document_type == "Extraction":
    st.write("Current schema used is: ")
    st.write("person_name, person_height, person_hair_color, dog_name, dog_breed")
    st.write("Use the following text as input:")
    st.write("Alex is 5 feet tall. Claudia is 1 feet taller Alex and jumps higher than him. Claudia is a brunette and Alex is blonde. Alex's dog Frosty is a labrador and likes to play hide and seek.")
    extraction_text = st.text_area(label="Enter text to extract from",
                                   placeholder="")
    if extraction_text:
        st.header("Original Text")
        st.write(extraction_text)
        response = extraction.extract(extraction_text)
        st.header("Extracted Information")
        st.write(response)
elif document_type == "Q&A From Documents":
    st.title("Ask a question based on multiple documents.")
    st.write("We have a bunch of documents related to history of India Cricket. You can ask any relevant question and the program will find the answer to it using the documents.")
    st.write("The program will load the books at runtime and query among them.")
    query = st.text_input(label="Please ask a question.",
                          placeholder="How many world cup's has India won?")
    if query:
        response = Q_and_A_docs.get_answers(query)
        st.write("Answer")
        answer = response["result"]
        st.write(answer)
elif document_type == "Q&A Evaluation":
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
    st.write("Evaluating Q&A prediction based on following examples: ")
    st.write(_examples)
    response = QAEvaluations.evaluate(_examples)
    examples = response["examples"]
    predictions = response["predictions"]
    graded_outputs = response["graded_outputs"]
    for i, eg in enumerate(examples):
        st.write(f"Example {i  + 1}:")
        st.write("Question: " + eg["question"])
        st.write("Real Answer: " + eg["answer"])
        st.write("Predicted Answer: " + predictions[i]["text"])
        st.write("Predicted Grade: " + graded_outputs[i]["text"])
elif document_type == "Query Latest Data":
    st.title("Query using an external tool and not chatGPT")
    st.write("This is useful when we want to retrieve any latest information but our model is not yet trained for it.")
    query = st.text_input(label="Please ask a question.",
                          placeholder="What day will Diwali be celebrated in 2023?")
    if query:
        response = SearchResults.get_answers(query)
        st.write("Answer")
        st.write(response)
elif document_type == "Query a Preloaded Book":
    st.title("Ask a question based on preloaded content")
    query = st.text_input(label="We have pre-loaded a book about basics of Computer Sciene. Please ask any relevant question and the program will find the answer to it using the uploaded book.'",
                          placeholder="What are the components of a computer?")
    if query:
        response = askabook.get_answers(query)
        st.write("Answer")
        st.write(response)
