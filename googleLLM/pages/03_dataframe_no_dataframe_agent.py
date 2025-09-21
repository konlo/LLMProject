import os
import sys
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# LLM (Gemini)
from langchain_google_genai import ChatGoogleGenerativeAI

# Streamlit 콜백/히스토리
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# ReAct + Python Tool
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain.agents import create_react_agent, AgentExecutor

# History wrapper
from langchain_core.runnables.history import RunnableWithMessageHistory

# Prompt
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 기타
from langchain.callbacks import StdOutCallbackHandler
import matplotlib.pyplot as plt

# CollectAllCallback (사용자 모듈)
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from modules.callbacks.collect_all import CollectAllCallback


# =============================
# App 시작
# =============================
load_dotenv()

st.set_page_config(page_title="DF Chatbot (Gemini)", page_icon="✨", layout="wide")
st.title("✨ DataFrame Chatbot (Gemini + LangChain)")
st.caption("Gemini + Python tool(ReAct)로 DataFrame을 분석합니다.")

# =============================
# 데이터 로드
# =============================
CSV_PATH = "/Users/najongseong/dataset/ncr_ride_bookings.csv"  # 필요 시 수정
df_ride_booking = pd.read_csv(CSV_PATH)

# =============================
# 환경 변수
# =============================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY가 설정되어 있지 않습니다. .env 또는 환경변수를 확인하세요.")
    st.stop()

# =============================
# LLM 세팅
# =============================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=GOOGLE_API_KEY,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    streaming=True,
)

# =============================
# 미리보기
# =============================
st.subheader("Preview")
st.dataframe(df_ride_booking.head(10), use_container_width=True)

# =============================
# Chat history
# =============================
history = StreamlitChatMessageHistory(key="lc_msgs:dfchat")

# =============================
# 프롬프트 (ReAct 필수 변수 포함)
# - {tools}, {tool_names}
# - chat_history는 MessagesPlaceholder (옵션)
# - agent_scratchpad는 문자열 슬롯으로만 받음 (assistant 라인)
# =============================
TOOL_HINT = (
    "When you use the python tool, return ONLY raw Python code "
    "(no backticks, no 'Action:' or 'Output:' lines, no prose). "
    "Put each statement on its own line. If you use 'pd', add 'import pandas as pd'. "
    "For charts, import matplotlib.pyplot as plt and end with 'plt.show()'. "
    "Keep plots around 600px width (e.g., figsize=(6,4), dpi=100)."
)


# df.head()를 system에 넣어주기 위해 문자열로 준비
df_head_txt = df_ride_booking.head().to_string(index=False)

react_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are working with a pandas dataframe named `df`.\n"
     "Tools available:\n{tools}\nUse only tool names from: {tool_names}.\n\n"
     "ALWAYS follow this exact format:\n"
     "Question: <restated question>\n"
     "Thought: <brief reasoning>\n"
     "Action: <ONE tool name from {tool_names}>\n"
     "Action Input: <valid python code with NO backticks>\n"
     "Observation: <tool result>\n"
     "(Repeat Thought/Action/Action Input/Observation as needed)\n"
     "Thought: I now know the final answer\n"
     "Final Answer: <answer>\n\n"
     "Example:\n"
     "Question: How many rows are in df?\n"
     "Thought: I should count rows using python.\n"
     "Action: python_repl_ast\n"
     "Action Input: len(df)\n"
     "Observation: 1234\n"
     "Thought: I now know the final answer\n"
     "Final Answer: 1234\n\n"
     "This is the result of print(df.head()):\n{df_head}\n"),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    # ReAct scratchpad는 문자열 슬롯
    ("assistant", "{agent_scratchpad}"),
]).partial(df_head=df_head_txt)


# =============================
# Python Tool (df/pd/plt 주입)
# =============================
pytool = PythonAstREPLTool(
    globals={
        "pd": pd,
        "plt": plt,
        "df": df_ride_booking,  # LLM이 'df'로 직접 접근
    }
)
tools = [pytool]

# =============================
# ReAct 에이전트
# =============================
react_runnable = create_react_agent(llm, tools, prompt=react_prompt)
agent = AgentExecutor(
    agent=react_runnable,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    return_intermediate_steps=True,
)

# =============================
# RunnableWithMessageHistory (자동 히스토리 주입/기록)
# =============================
agent_with_history = RunnableWithMessageHistory(
    agent,
    lambda session_id: history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# =============================
# UI
# =============================
st.write("---")
user_q = st.chat_input("Ask a question about your data (예: '상위 5개 항목의 TBW를 보여줘')")

if user_q:
    # 좌/우 컬럼
    left, right = st.columns([1, 1])
    with left:
        st.subheader("실시간 실행 로그")
        st_cb = StreamlitCallbackHandler(st.container())

    collector = CollectAllCallback()

    with st.spinner("Thinking with Gemini..."):
        result = agent_with_history.invoke(
            {"input": user_q},  # 질문만 전달 (가이드는 system에 이미 포함)
            {
                "callbacks": [st_cb, collector, StdOutCallbackHandler()],
                "configurable": {"session_id": "konlo_ssid"},  # 세션 ID 필수
            }
        )

    st.success("Done.")

    # 최종 답변
    final = result.get("output", result)
    with right:
        st.subheader("Answer")
        st.write(final)

        # matplotlib figure 자동 수집/표시 (600px 근사 강제)
        figs = [plt.figure(n) for n in plt.get_fignums()]
        for f in figs:
            f.set_size_inches(6, 4)
            f.set_dpi(100)
            st.pyplot(f, use_container_width=True)
        plt.close("all")

    # =============================
    # 중간 스텝 & 이벤트 타임라인
    # =============================
    st.write("---")
    st.subheader("intermediate_steps (툴 실행 상세)")
    steps = result.get("intermediate_steps", [])
    if steps:
        for i, (action, observation) in enumerate(steps, 1):
            with st.expander(f"Step {i}: {action.tool}"):
                st.markdown("**tool_input (실행된 코드/질의)**")
                st.code(str(action.tool_input))
                st.markdown("**observation (툴 결과)**")
                st.write(observation)
                st.markdown("**raw log**")
                st.code(action.log or "")
    else:
        st.info("intermediate_steps가 비어 있습니다. 콜백 이벤트 타임라인에서 세부 과정을 확인하세요.")

    st.write("---")
    st.subheader("콜백 이벤트 타임라인 (프롬프트/툴 I/O)")
    for j, e in enumerate(collector.events, 1):
        with st.expander(f"Event {j}: {e.get('type')}"):
            if e.get("type") == "llm_start":
                prompts = e.get("prompts", [])
                for k, p in enumerate(prompts, 1):
                    st.markdown(f"**Prompt {k}**")
                    st.code(p)
            else:
                st.write(e)
