from tempfile import NamedTemporaryFile

import streamlit as st
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from tools import ImageCaptionTool, ObjectDetectionTool





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
st.title('Ask a question to an image')

# set header
# st.header("Please upload an image")

# upload file
file = st.file_uploader("Please upload an image", type=["jpeg", "jpg", "png"])

if file:
    # display image
    st.image(file, use_column_width=True)

    # text input
    user_question = st.text_input('Ask a question about your image:')

    ##############################
    ### compute agent response ###
    ##############################
    with NamedTemporaryFile(mode='w+b',dir='.') as f:
        f.write(file.getbuffer())
        image_path = f.name



        # write agent response
        if user_question and user_question != "":
            with st.spinner(text="In progress..."):
                response = agent.run(f'{user_question}, this is the image path: {image_path}')
                st.write(response)