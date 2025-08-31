import google.generativeai as genai
from pandasai import Agent
import pandas as pd
genai.configure(api_key="AIzaSyAXq7IKW6QToj9MOeTA3uvADiAL3t8vv3c")
google_llm = genai.GenerativeModel("gemini-2.0-flash")



import streamlit as st

from pandasai.llm import LLM

class GeminiLLM(LLM):
    def __init__(self, api_key):
        self.api_key = api_key
        # Initialize any other necessary attributes
    @property
    def type(self) -> str:
        # PandasAI가 인식할 임의의 식별자(문자열) 지정
        # (내장 enum이 아니어도 문자열이면 동작하도록 되어 있습니다)
        return "google_gemini_custom"

    def call(self, instruction, context=None):
        # Implement the logic to call the Gemini API and return the response
        pass



if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_prompt = st.chat_input("Ask MIS Agent...")

if user_prompt:
# add user's message to chat history and display it
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role":"user","content": user_prompt})

data = {
    'product': ['노트북', '마우스', '키보드', '모니터', '노트북', '마우스'],
    'category': ['전자제품', '전자제품', '전자제품', '전자제품', '전자제품', '전자제품'],
    'price': [1200000, 35000, 60000, 450000, 1500000, 40000],
    'quantity': [5, 12, 8, 3, 2, 15],
    'sale_date': ['2023-01-10', '2023-01-11', '2023-01-12', '2023-01-13', '2023-02-01', '2023-02-02']
}
sales_by_partner = pd.DataFrame(data)

google_llm = GeminiLLM(api_key="AIzaSyAXq7IKW6QToj9MOeTA3uvADiAL3t8vv3c")
agent = Agent(sales_by_partner, config={"llm": google_llm})

# agent = Agent(sales_by_partner, config={"llm": google_llm})
assistant_response = agent.chat(user_prompt)


st.session_state.chat_history.append({"role":"assistant", "content": assistant_response})

# display pandasai response
with st.chat_message("assistant"):
    assistant_response
