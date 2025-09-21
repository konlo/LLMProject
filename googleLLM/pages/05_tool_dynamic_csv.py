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
# CSV 선택 (사이드바) → df_A
# =============================
DATA_DIR = "/Users/najongseong/dataset"  # 기본 데이터 디렉터리
DEFAULT_CSV = os.path.join(DATA_DIR, "telemetry_report.csv")  # df_A 기본 파일
DFB_DEFAULT = os.path.join(DATA_DIR, "telemetry_raw.csv")     # df_B 기본 파일

if "csv_path" not in st.session_state:
    st.session_state["csv_path"] = DEFAULT_CSV

with st.sidebar:
    st.markdown("### 📄 CSV Path (df_A)")
    new_path = st.text_input("CSV file path for df_A", value=st.session_state["csv_path"])
    if st.button("Load CSV for df_A"):
        st.session_state["csv_path"] = new_path.strip()
        st.rerun()
    st.caption(f"df_B 기본 가정 파일: `{DFB_DEFAULT}` (load_df_b() 파일명 생략 시 사용)")

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
st.write(f"**Loaded CSV for df_A:** `{CSV_PATH}`")
st.dataframe(df.head(10), use_container_width=True)

# =============================
# Chat history
# =============================
history = StreamlitChatMessageHistory(key="lc_msgs:dfchat")

# =============================
# Prompt (ReAct 형식 + Tool 라우팅 가이드 + 파싱 복구 지침)
# =============================
df_head_txt = df.head().to_string(index=False)

react_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are working with pandas dataframes. The main dataframe is `df_A` (alias: df). "
     "You may also load a related dataframe as `df_B` for root-cause analysis.\n\n"
     "Tools available:\n{tools}\nUse only tool names from: {tool_names}.\n"
     "If you use sql_on_dfs, available tables are df_A and (if loaded) df_B.\n\n"
     "Tool routing guide:\n"
     "- schema/summary → describe_columns or describe_columns_on\n"
     "- load file → load_loading_csv or load_df_b\n"
     "- SQL/join/aggregation → sql_on_dfs\n"
     "- custom compute/plots → python_repl_ast\n"
     "- suggest join keys → propose_join_keys\n"
     "- align timestamps to buckets → align_time_buckets\n\n"
     "Parsing recovery:\n"
     "- If you output 'Action:' without 'Action Input:', immediately continue with only 'Action Input: <...>'.\n"
     "- Do not wrap code in backticks.\n"
     "- Keep Action Input minimal, valid, and executable.\n\n"
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
     "Question: How many rows are in df_A?\n"
     "Thought: I should count rows using python.\n"
     "Action: python_repl_ast\n"
     "Action Input: len(df_A)\n"
     "Observation: 1234\n"
     "Thought: I now know the final answer\n"
     "Final Answer: 1234\n\n"
     f"This is the result of print(df_A.head()):\n{df_head_txt}\n"),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    ("assistant", "{agent_scratchpad}"),
])

# =============================
# Tools 정의
# =============================

# 1) Python 실행 툴: df_A/df_B/df/loading_df 노출
pytool = PythonAstREPLTool(
    globals={
        "pd": pd,
        "plt": plt,
        "df": df,      # 호환성 유지 (df == df_A)
        "df_A": df,    # 메인 데이터셋
        "df_B": None,  # 보조 데이터셋 (툴로 로드)
        "loading_df": None,  # ad-hoc 분석용
    },
    name="python_repl_ast",
    description="Execute Python on df_A/df_B with pandas/matplotlib. 'df' aliases df_A."
)

# 2) 동적 CSV 로드 (loading_df)
@tool
def load_loading_csv(filename: str) -> str:
    """Load a CSV from DATA_DIR into 'loading_df' (for ad-hoc analysis). Pass only file name (e.g., 'loading_test.csv')."""
    path = os.path.join(DATA_DIR, filename)
    try:
        new_df = pd.read_csv(path)
    except Exception as e:
        return f"Failed to load {path}: {e}"
    pytool.globals["loading_df"] = new_df
    preview = new_df.head(10).to_markdown(index=False)
    shape = f"{new_df.shape[0]} rows x {new_df.shape[1]} cols"
    return f"Loaded {filename} from {DATA_DIR} into loading_df\nShape: {shape}\n\nPreview (head):\n{preview}"

# 3) 컬럼 요약 (loading_df 우선 → 없으면 df_A)
@tool
def describe_columns(cols: str = "") -> str:
    """
    Describe selected columns (comma-separated).
    Uses 'loading_df' if available; otherwise uses 'df_A'.
    """
    if "loading_df" in pytool.globals and pytool.globals["loading_df"] is not None:
        current_df = pytool.globals["loading_df"]
        source = "loading_df"
    else:
        current_df = pytool.globals.get("df_A", df) if pytool.globals.get("df_A") is not None else df
        source = "df_A"

    use_cols = [c.strip() for c in cols.split(",") if c.strip()] or list(current_df.columns)
    missing = [c for c in use_cols if c not in current_df.columns]
    if missing:
        return f"Missing columns: {missing}\nAvailable: {list(current_df.columns)}"

    desc = current_df[use_cols].describe(include="all").transpose()
    shape = f"{current_df.shape[0]} rows x {current_df.shape[1]} cols"
    return f"[source={source} | shape={shape}]\n\n" + desc.to_markdown()

# 4) 플롯 ZIP
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

# 5) df_B 로더 (파일명 생략 시 telemetry_raw.csv 기본 사용)
@tool
def load_df_b(filename: str = "") -> str:
    """
    Load a CSV into 'df_B'. If 'filename' is empty, defaults to 'telemetry_raw.csv' under DATA_DIR.
    Example: load_df_b()  # uses telemetry_raw.csv
             load_df_b('other_raw.csv')
    """
    if not filename:
        path = DFB_DEFAULT
        show_name = os.path.basename(DFB_DEFAULT)
    else:
        path = os.path.join(DATA_DIR, filename) if not os.path.isabs(filename) else filename
        show_name = os.path.basename(path)

    try:
        new_df = pd.read_csv(path)
    except Exception as e:
        return f"Failed to load df_B from {path}: {e}"
    pytool.globals["df_B"] = new_df
    preview = new_df.head(10).to_markdown(index=False)
    return f"Loaded df_B from '{show_name}' (full: {path}) with shape {new_df.shape}\n\nPreview:\n{preview}"

# 6) df_A / df_B 대상 지정 요약
@tool
def describe_columns_on(target: str = "A", cols: str = "") -> str:
    """
    Describe columns from df_A or df_B.
    target: 'A' or 'B'
    cols: comma-separated; empty -> all
    """
    t = (target or "A").strip().upper()
    if t == "B":
        if "df_B" not in pytool.globals or pytool.globals["df_B"] is None:
            return "df_B is not loaded. Use load_df_b() first."
        current_df = pytool.globals["df_B"]
        source = "df_B"
    else:
        current_df = pytool.globals.get("df_A", df) if pytool.globals.get("df_A") is not None else df
        source = "df_A"

    use_cols = [c.strip() for c in cols.split(",") if c.strip()] or list(current_df.columns)
    missing = [c for c in use_cols if c not in current_df.columns]
    if missing:
        return f"Missing columns: {missing}\nAvailable: {list(current_df.columns)}"

    desc = current_df[use_cols].describe(include="all").transpose()
    shape = f"{current_df.shape[0]} rows x {current_df.shape[1]} cols"
    return f"[source={source} | shape={shape}]\n\n" + desc.to_markdown()

# 7) df_A/df_B SQL 질의 (DuckDB)
@tool
def sql_on_dfs(query: str) -> str:
    """
    Run DuckDB SQL over df_A and (if loaded) df_B.
    Tables: df_A, df_B
    Example:
      SELECT a.key, a.metric, b.cause
      FROM df_A a LEFT JOIN df_B b ON a.key = b.key
      WHERE a.metric > 100
      ORDER BY a.metric DESC
      LIMIT 20
    """
    try:
        duckdb.register("df_A", pytool.globals.get("df_A", df) if pytool.globals.get("df_A") is not None else df)
        if pytool.globals.get("df_B") is not None:
            duckdb.register("df_B", pytool.globals["df_B"])
        out = duckdb.sql(query).df()
        return out.head(200).to_markdown(index=False)
    except Exception as e:
        return f"SQL error: {e}"

# 8) 조인 키 후보 추천
@tool
def propose_join_keys() -> str:
    """Suggest join key candidates between df_A and df_B by intersecting column names and compatible dtypes."""
    import numpy as np
    A = pytool.globals.get("df_A") if pytool.globals.get("df_A") is not None else df
    B = pytool.globals.get("df_B")
    if B is None:
        return "df_B is not loaded."

    def dtype_sig(series):
        t = str(series.dtype)
        if "int" in t: return "int"
        if "float" in t: return "float"
        if "datetime" in t or "date" in t or "time" in t: return "datetime"
        return "str"

    pairs = []
    for c in A.columns:
        if c in B.columns and dtype_sig(A[c]) == dtype_sig(B[c]):
            pairs.append((c, dtype_sig(A[c])))

    if not pairs:
        return "No obvious same-name & same-type keys. Consider mapping tables or casting."

    md = "| key | dtype |\n|---|---|\n" + "\n".join([f"| {k} | {t} |" for k, t in pairs])
    return f"Candidate join keys (same name & type):\n{md}"

# 9) 타임버킷 정렬
@tool
def align_time_buckets(target: str = "A", column: str = "ts", freq: str = "H") -> str:
    """
    Resample time-like column to buckets and store as df_A_bucketed or df_B_bucketed.
    target: 'A' or 'B'; column must be timestamp-like; freq like 'H','D','15min'
    """
    import pandas as pd
    t = (target or "A").strip().upper()
    cur = None
    if t == "B":
        cur = pytool.globals.get("df_B")
        if cur is None:
            return "df_B is not loaded."
    else:
        cur = pytool.globals.get("df_A") if pytool.globals.get("df_A") is not None else df

    if column not in cur.columns:
        return f"Column '{column}' not in df_{t}. Available: {list(cur.columns)}"

    tmp = cur.copy()
    tmp[column] = pd.to_datetime(tmp[column], errors="coerce")
    bucket_col = f"{column}_bucket"
    tmp[bucket_col] = tmp[column].dt.to_period(freq).dt.to_timestamp()

    key = f"df_{t}_bucketed"
    pytool.globals[key] = tmp
    prev = tmp[[bucket_col]].head().to_markdown(index=False)
    return f"Created {key} with '{bucket_col}' at freq={freq}.\nPreview:\n{prev}"

tools = [
    pytool,
    load_loading_csv,
    describe_columns,
    save_plots_zip,
    load_df_b,             # 파일명 없으면 telemetry_raw.csv 사용
    describe_columns_on,
    sql_on_dfs,
    propose_join_keys,
    align_time_buckets,
]

# =============================
# ReAct Agent
# =============================
react_runnable = create_react_agent(llm, tools, prompt=react_prompt)
agent = AgentExecutor(
    agent=react_runnable,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True,
    handle_parsing_errors=(
        "FORMAT REMINDER: After 'Action:' you MUST output\n"
        "Action Input: <valid code with NO backticks>\n"
        "If you omitted it, continue by providing Action Input only."
    ),
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
user_q = st.chat_input(
    "Ask about your data (예: 'load_df_b() 후 describe_columns_on target=B', "
    "'propose_join_keys', 'sql_on_dfs 로 df_A/df_B 조인', "
    "'align_time_buckets target=A column=created_at freq=H')"
)

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
