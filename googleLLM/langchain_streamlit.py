import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv


# LangChain + Gemini
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

load_dotenv()  # .env 로드

# Check if GOOGLE_API_KEY is set in environment variables
if "GOOGLE_API_KEY" in os.environ:
    print("GOOGLE_API_KEY is set.")
    print("Value:", os.environ["GOOGLE_API_KEY"])
else:
    print("GOOGLE_API_KEY is not set.")

st.set_page_config(page_title="DF Chatbot (Gemini)", page_icon="✨")
st.title("✨ DataFrame Chatbot (Gemini + LangChain)")
st.caption("Uses Google Gemini through LangChain's `create_pandas_dataframe_agent`.")

df_ride_booking = pd.read_csv(
    # "https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv"
    "/Users/najongseong/dataset/ncr_ride_bookings.csv"
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

st.subheader("Preview")
st.dataframe(df_ride_booking.head(10))


# Create the agent
agent = create_pandas_dataframe_agent(
    llm=llm,
    df=df_ride_booking,
    verbose=True,
    allow_dangerous_code=True,
)

st.write("---")
user_q = st.text_input("Ask a question about your data (예: '상위 5개 항목의 TBW를 보여줘')")

go = st.button("Run ▶️")
if go and user_q.strip():
    st_cb = StreamlitCallbackHandler(st.container())
    with st.spinner("Thinking with Gemini..."):
        result = agent.invoke({"input": user_q}, {"callbacks": [st_cb]})
    st.success("Done.")
    final = result.get("output", result)
    st.subheader("Answer")
    st.write(final)

    import matplotlib.pyplot as plt
    figs = [plt.figure(n) for n in plt.get_fignums()]
    for f in figs:
        st.pyplot(f)
    plt.close("all")    
elif go:
    st.warning("Please enter a question.")
