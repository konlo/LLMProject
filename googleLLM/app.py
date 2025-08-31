from pandasai.llm.local_llm import LocalLLM
import streamlit as st
import pandas as pd

from pandasai import SmartDataframe


model = LocalLLM(
    api_base="http://localhost:11434/v1",
    model="gemma3:1b"
)

st.title("PandasAI Streamlit App")
upload_file = st.file_uploader("Upload your CSV file", type=["csv"])

if upload_file is not None: 
    data = pd.read_csv(upload_file)
    st.write(data.head(3))

    df = SmartDataframe(data, config={"llm": model})
    prompt = st.text_input("Enter your prompt")

    if st.button("Submit"):
        if prompt:
            with st.spinner("Generating response..."):
                st.write(df.chat(prompt))
        else:
            st.write("Please enter a prompt.")