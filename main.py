import streamlit as st
import summarize

st.title("Summarize document")

file = st.file_uploader(label="Upload document", accept_multiple_files=False, type=".txt")

if file:
    response = summarize.summarize(file)
    st.header("Summary")
    st.write(response)