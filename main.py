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

from tools import ImageCaptionTool, ObjectDetectionTool

document_type = st.sidebar.selectbox("Choose an Example", ("Summarize a Document","Describe an Image","Extraction", "Q&A From Documents", "Q&A Evaluation"))

if document_type == "Summarize a Document":
    st.title("Summarize document")

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
        openai_api_key='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',
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
    file = st.file_uploader("Please upload an image", type=["jpeg", "jpg", "png"])
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
            response1 = agent.run(f'{user_question_1}, this is the image path: {file_path}')
            st.write("The description of the image is: " + response1)
            response2 = agent.run(f'{user_question_2}, this is the image path: {file_path}')
            st.write(response2)

            # Clean up the temporary directory
            temp_dir.cleanup()
elif document_type == "Extraction":
    st.write("Current schema used is: ")
    st.write("person_name, person_height, person_hair_color, dog_name, dog_breed")
    extraction_text = st.text_area(label="Enter text to extract from",
                                   placeholder="Alex is 5 feet tall. Claudia is 1 feet taller Alex and jumps higher than him. Claudia is a brunette and Alex is blonde. Alex's dog Frosty is a labrador and likes to play hide and seek.")
    if extraction_text:
        st.header("Original Text")
        st.write(extraction_text)
        response = extraction.extract(extraction_text)
        st.header("Extracted Information")
        st.write(response)
elif document_type == "Q&A From Documents":
    query = st.text_input(label="Please ask a question.",
                          placeholder="What is compro technologies?")
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