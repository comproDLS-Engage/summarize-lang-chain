import streamlit as st
import summarize
import tempfile
import os
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from tools import ImageCaptionTool, ObjectDetectionTool

document_type = st.sidebar.selectbox("Pick a File Type", ("Text","Image"))

if document_type == "Text":
    st.title("Summarize document")

    file = st.file_uploader(label="Upload document",
                        accept_multiple_files=True, type=".txt")

    if file:
        response = summarize.summarize(file)
        for essay in response:
            st.header("Summary of " + essay["name"])
            st.write(essay["summary"])
elif document_type == "Image":
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
    st.title('Generate the summary and objects of the image')

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
            st.write(response1)
            response2 = agent.run(f'{user_question_2}, this is the image path: {file_path}')
            st.write(response2)

            # Clean up the temporary directory
            temp_dir.cleanup()

