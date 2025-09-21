import os, streamlit as st, pandas as pd, pandasai as pai
from pandasai_litellm.litellm import LiteLLM
os.environ["GOOGLE_API_KEY"] = "AIzaSyAXq7IKW6QToj9MOeTA3uvADiAL3t8vv3c"
# os.environ["GEMINI_API_KEY"] = os.getenv("GOOGLE_API_KEY")
pai.config.set({"llm": LiteLLM(model="gemini/gemini-2.0-flash")})

df = pai.DataFrame({"Name":["Alice","Bob","Charlie"], "Age":[25,30,35]})
st.write(df.chat("Who is the oldest person?"))
