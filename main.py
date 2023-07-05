import streamlit as st
import summarize

st.title("Summarize document")

file = st.file_uploader(label="Upload document",
                        accept_multiple_files=True, type=".txt")

if file:
    response = summarize.summarize(file)
    for essay in response:
        st.header("Summary of " + essay["name"])
        st.write(essay["summary"])
