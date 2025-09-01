import os
import pandas as pd
import streamlit as st
from pandasai import SmartDataframe, config
from pandasai_litellm.litellm import LiteLLM
import matplotlib.pyplot as plt
from PIL import Image

# 1. API 키 설정 (실제 코드에서는 환경변수로 받는 것이 안전)
os.environ["GOOGLE_API_KEY"] = "AIzaSyAXq7IKW6QToj9MOeTA3uvADiAL3t8vv3c"
os.environ["GEMINI_API_KEY"] = os.getenv("GOOGLE_API_KEY")  # LiteLLM이 이 이름을 사용

# 2. PandasAI 전역 설정: LLM 연결 + 출력 경로 설정
config.set({
    "llm": LiteLLM(model="gemini/gemini-2.0-flash"),
    "save_charts": True,
    "save_charts_path": "./charts",
    "enable_cache": False  # 캐시 비활성화로 일관된 결과 확보
})

# charts 디렉토리 생성
os.makedirs("./charts", exist_ok=True)

# 3. Streamlit UI
st.title("PandasAI Streamlit App with Visualization")
upload_file = st.file_uploader("Upload your CSV file", type=["csv"])

if upload_file is not None: 
    data = pd.read_csv(upload_file)
    st.write("### Data Preview:")
    st.write(data.head(3))

    # 4. SmartDataframe 생성
    df = SmartDataframe(data)

    # 5. 사용자 프롬프트 입력
    prompt = st.text_input("Enter your prompt (try asking for charts!)")
    
    # 예시 프롬프트 제안
    st.write("**Example prompts:**")
    st.write("- Create a bar chart of sales by category")
    st.write("- Show me a line plot of revenue over time")
    st.write("- Visualize the distribution of prices")

    if prompt:
        # 6. 자연어 쿼리 실행
        with st.spinner("Thinking..."):
            try:
                answer = df.chat(prompt)
                
                st.markdown("#### Answer:")
                
                # 답변이 문자열인 경우 (일반적인 데이터 분석 결과)
                if isinstance(answer, str):
                    st.write(answer)
                
                # 답변이 DataFrame인 경우
                elif isinstance(answer, pd.DataFrame):
                    st.dataframe(answer)
                
                # 다른 타입의 답변
                else:
                    st.write(answer)
                
                # 차트가 생성되었는지 확인하고 표시
                charts_dir = "./charts"
                if os.path.exists(charts_dir):
                    chart_files = [f for f in os.listdir(charts_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
                    if chart_files:
                        st.markdown("#### Generated Chart:")
                        # 가장 최근에 생성된 차트 표시
                        latest_chart = max([os.path.join(charts_dir, f) for f in chart_files], 
                                         key=os.path.getctime)
                        
                        # 이미지 표시
                        image = Image.open(latest_chart)
                        st.image(image, caption="Generated Visualization", use_column_width=True)
                        
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.write("Try rephrasing your question or ask for a different type of analysis.")

# 7. 사이드바에 도움말 정보
with st.sidebar:
    st.markdown("### Tips:")
    st.write("- Upload a CSV file to get started")
    st.write("- Ask questions in natural language")
    st.write("- Request specific chart types (bar, line, scatter, etc.)")
    st.write("- Charts will be automatically displayed below your answer")