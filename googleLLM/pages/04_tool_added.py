# 요약: Streamlit 앱이 CSV를 로드/미리보기하고, Gemini(LangChain) ReAct 에이전트를 구성해 `df`를 파이썬 도구로 분석합니다.
# 요약: 툴 구성 — `python_repl_ast`, `load_loading_csv`(보조 DataFrame), `describe_columns`, `save_plots_zip`; RunnableWithMessageHistory로 히스토리 유지.
# 요약: 실시간 실행 로그·최종 답변·플롯을 표시하고, intermediate_steps와 CollectAllCallback 타임라인으로 실행 내역을 시각화합니다.
# 작성자: konlona · 날짜: 2025-09-21 (KST)


import os
import sys
import io
import zipfile
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
from langchain_core.tools import tool  # @tool 데코레이터

# 기타
from langchain.callbacks import StdOutCallbackHandler
import matplotlib.pyplot as plt
import duckdb

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
# CSV 선택 (사이드바)
# =============================
DEFAULT_CSV = "/Users/najongseong/dataset/ncr_ride_bookings.csv"
if "csv_path" not in st.session_state:
    st.session_state["csv_path"] = DEFAULT_CSV

with st.sidebar:
    st.markdown("### 📄 CSV Path")
    new_path = st.text_input("CSV file path", value=st.session_state["csv_path"])
    if st.button("Load CSV"):
        st.session_state["csv_path"] = new_path.strip()
        st.rerun()

CSV_PATH = st.session_state["csv_path"]
try:
    df = pd.read_csv(CSV_PATH)
except Exception as e:
    st.error(f"CSV 로드 실패: {CSV_PATH}\n{e}")
    st.stop()

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
# Preview
# =============================
st.subheader("Preview")
st.write(f"**Loaded CSV:** `{CSV_PATH}`")
st.dataframe(df.head(10), use_container_width=True)

# =============================
# Chat history
# =============================
history = StreamlitChatMessageHistory(key="lc_msgs:dfchat")

# =============================
# Prompt (ReAct 형식 + tool 안내)
# =============================
df_head_txt = df.head().to_string(index=False)

react_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are working with a pandas dataframe named `df`.\n"
     "Tools available:\n{tools}\nUse only tool names from: {tool_names}.\n\n"
     "ALWAYS follow this exact format:\n"
     "Question: <restated question>\n"
     "Thought: <brief reasoning>\n"
     "Action: <ONE tool name from {tool_names}>\n"
     "Action Input: <valid code with NO backticks>\n"
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
     f"This is the result of print(df.head()):\n{df_head_txt}\n"),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    ("assistant", "{agent_scratchpad}"),
])

# =============================
# Tools 정의
# =============================

# 1) Python 실행 툴
pytool = PythonAstREPLTool(
    globals={"pd": pd, "plt": plt, "df": df},
    name="python_repl_ast",
    description="Execute Python code using pandas/matplotlib on df."
)

# 2) CSV 로드 툴
# 2) CSV 로드 툴
DATA_DIR = "/Users/najongseong/dataset"

@tool
def load_loading_csv(filename: str) -> str:
    """Load a CSV file from /Users/najongseong/dataset into 'loading_df' for analysis.
    Pass only the file name (e.g., 'loading_test.csv').
    """
    path = os.path.join(DATA_DIR, filename)
    try:
        new_df = pd.read_csv(path)
    except Exception as e:
        return f"Failed to load {path}: {e}"
    pytool.globals["loading_df"] = new_df
    preview = new_df.head(10).to_markdown(index=False)
    shape = f"{new_df.shape[0]} rows x {new_df.shape[1]} cols"
    return f"Loaded {filename} from {DATA_DIR}\nShape: {shape}\n\nPreview (head):\n{preview}"


# 3) 컬럼 요약 툴 (동적 df 사용)
@tool
def describe_columns(cols: str = "") -> str:
    """
    Describe selected columns (comma-separated).
    If 'loading_df' exists, use it; otherwise use 'df'.
    """
    # None 비교를 명시적으로!
    if "loading_df" in pytool.globals and pytool.globals["loading_df"] is not None:
        current_df = pytool.globals["loading_df"]
        source = "loading_df"
    else:
        current_df = df
        source = "df"

    use_cols = [c.strip() for c in cols.split(",") if c.strip()] or current_df.columns.tolist()
    missing = [c for c in use_cols if c not in current_df.columns]
    if missing:
        return f"Missing columns: {missing}\nAvailable: {list(current_df.columns)}"

    desc = current_df[use_cols].describe(include="all").transpose()
    shape = f"{current_df.shape[0]} rows x {current_df.shape[1]} cols"
    return f"[source={source} | shape={shape}]\n\n" + desc.to_markdown()


# 4) 플롯 ZIP 툴
@tool
def save_plots_zip() -> str:
    """Zip all current matplotlib figures. Use after plotting with python_repl_ast."""
    figs = [plt.figure(n) for n in plt.get_fignums()]
    if not figs:
        return "No figures to save."
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, f in enumerate(figs, 1):
            img = io.BytesIO()
            f.savefig(img, format="png", dpi=100, bbox_inches="tight")
            img.seek(0)
            zf.writestr(f"plot_{i}.png", img.read())
    return f"Zipped {len(figs)} plots. Bytes={len(buf.getvalue())}"

tools = [pytool, load_loading_csv, describe_columns, save_plots_zip]

# =============================
# ReAct Agent
# =============================
react_runnable = create_react_agent(llm, tools, prompt=react_prompt)
agent = AgentExecutor(
    agent=react_runnable,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True,
    handle_parsing_errors=True,
)

# =============================
# RunnableWithMessageHistory
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
user_q = st.chat_input("Ask about your data (예: '상위 5개 TBW', 'load_loading_csv로 titanic.csv 불러와서 describe_columns')")

if user_q:
    left, right = st.columns([1, 1])
    with left:
        st.subheader("실시간 실행 로그")
        st_cb = StreamlitCallbackHandler(st.container())

    collector = CollectAllCallback()

    with st.spinner("Thinking with Gemini..."):
        result = agent_with_history.invoke(
            {"input": user_q},
            {
                "callbacks": [st_cb, collector, StdOutCallbackHandler()],
                "configurable": {"session_id": "konlo_ssid"},
            }
        )

    st.success("Done.")

    final = result.get("output", result)
    with right:
        st.subheader("Answer")
        st.write(final)

        # 플롯 렌더링
        figs = [plt.figure(n) for n in plt.get_fignums()]
        for f in figs:
            f.set_size_inches(6, 4)
            f.set_dpi(100)
            st.pyplot(f, use_container_width=True)
        plt.close("all")

    # intermediate steps
    st.write("---")
    st.subheader("intermediate_steps (툴 실행 상세)")
    steps = result.get("intermediate_steps", [])
    if steps:
        for i, (action, observation) in enumerate(steps, 1):
            with st.expander(f"Step {i}: {action.tool}"):
                st.markdown("**tool_input**")
                st.code(str(action.tool_input))
                st.markdown("**observation**")
                st.write(observation)
    else:
        st.info("intermediate_steps 비어 있음")

    # callback timeline
    st.write("---")
    st.subheader("콜백 이벤트 타임라인")
    for j, e in enumerate(collector.events, 1):
        with st.expander(f"Event {j}: {e.get('type')}"):
            if e.get("type") == "llm_start":
                prompts = e.get("prompts", [])
                for k, p in enumerate(prompts, 1):
                    st.markdown(f"**Prompt {k}**")
                    st.code(p)
            else:
                st.write(e)
