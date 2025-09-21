import os
import pandas as pd
import streamlit as st
from pandasai import SmartDataframe, config
from pandasai_litellm.litellm import LiteLLM

# 1. API 키 설정 (실제 코드에서는 환경변수로 받는 것이 안전)
os.environ["GOOGLE_API_KEY"] = "AIzaSyAXq7IKW6QToj9MOeTA3uvADiAL3t8vv3c"
os.environ["GEMINI_API_KEY"] = os.getenv("GOOGLE_API_KEY")  # LiteLLM이 이 이름을 사용

# 2. PandasAI 전역 설정: LLM 연결
config.set({
    "llm": LiteLLM(model="gemini/gemini-2.0-flash")
})

# 3. Streamlit UI
st.title("PandasAI Streamlit App")
upload_file = st.file_uploader("Upload your CSV file", type=["csv"])

if upload_file is not None: 
    data = pd.read_csv(upload_file)
    st.write(data.head(3))

    # 4. SmartDataframe 생성 (LiteLLM은 이미 전역 설정됨)
    df = SmartDataframe(data)  # config={"llm": ...} 생략 가능

    # 5. 사용자 프롬프트 입력
    prompt = st.text_input("Enter your prompt")

    if prompt:
        # 6. 자연어 쿼리 실행
        with st.spinner("Thinking..."):
            answer = df.chat(prompt)
        st.markdown("#### Answer:")
        st.write(answer)
