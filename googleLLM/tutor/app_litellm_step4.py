#2025.09.02 konlo.na
# image 출력하도록 수정함. 
# 여기에서 SmartDataFrame은 사용하지 않느 것으로 수정 
# 2025.09.02 konlo.na
# SmartDataframe 제거 + df.chat() 사용
# 차트 자동 저장 후 최신 이미지 표시

import os
import pandas as pd
import streamlit as st
import pandasai as pai
from pandasai import config
from pandasai_litellm.litellm import LiteLLM
from PIL import Image

# 1) API 키 (실서비스는 환경변수 권장)
os.environ["GOOGLE_API_KEY"] = "AIzaSyAXq7IKW6QToj9MOeTA3uvADiAL3t8vv3c"
os.environ["GEMINI_API_KEY"] = os.getenv("GOOGLE_API_KEY")  # LiteLLM이 이 이름을 사용

# 2) PandasAI 전역 설정: LLM + 차트 저장 경로
config.set({
    "llm": LiteLLM(model="gemini/gemini-2.0-flash"),
    "save_charts": True,
    "save_charts_path": "./charts",
    "enable_cache": False,  # 캐시 비활성화(테스트/일관성)
    # 선택: 모델이 SQL 경로를 따르도록 힌트 주기
    "instruction": (
        "When analyzing the dataframe, prefer generating SQL and execute it via "
        "`execute_sql_query`. Use code to save plots to files so they appear in ./charts."
    ),
})

# charts 디렉토리 생성
os.makedirs("./charts", exist_ok=True)

# 3) Streamlit UI
st.title("PandasAI Streamlit App with Visualization (No SmartDataframe)")
upload_file = st.file_uploader("Upload your CSV file", type=["csv"])

if upload_file is not None:
    df = pd.read_csv(upload_file)  # << 그냥 pandas DataFrame
    st.write("### Data Preview:")
    st.write(df.head(3))

    prompt = st.text_input("Enter your prompt (try asking for charts!)")
    st.write("**Example prompts:**")
    st.write("- Create a bar chart of sales by category")
    st.write("- Show me a line plot of revenue over time")
    st.write("- Visualize the distribution of prices")

    if prompt:
        with st.spinner("Thinking..."):
            try:
                # PandasAI 3.x: pandas.DataFrame에 바로 .chat() 가능
                # (SQL-first 검증 때문에, 자연어에 'use SQL via execute_sql_query'를 넣어두면 안전)
                answer = df.chat(
                    prompt + "\n(Use SQL via execute_sql_query and save any plots to files.)"
                )

                st.markdown("#### Answer:")
                if isinstance(answer, pd.DataFrame):
                    st.dataframe(answer)
                else:
                    st.write(answer)

                # 생성된 최신 차트 표시
                charts_dir = "./charts"
                if os.path.exists(charts_dir):
                    chart_files = [
                        f for f in os.listdir(charts_dir)
                        if f.lower().endswith((".png", ".jpg", ".jpeg"))
                    ]
                    if chart_files:
                        latest_chart = max(
                            (os.path.join(charts_dir, f) for f in chart_files),
                            key=os.path.getctime
                        )
                        image = Image.open(latest_chart)
                        st.markdown("#### Generated Chart:")
                        st.image(image, caption="Generated Visualization", use_column_width=True)

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.caption("If you see `ExecuteSQLQueryNotUsed`, try rephrasing to explicitly ask "
                           "the model to use SQL via `execute_sql_query`.")

# 4) 사이드바 도움말
with st.sidebar:
    st.markdown("### Tips:")
    st.write("- Upload a CSV file to get started")
    st.write("- Ask questions in natural language")
    st.write("- Request specific chart types (bar, line, scatter, etc.)")
    st.write("- Charts will be automatically displayed if created")
