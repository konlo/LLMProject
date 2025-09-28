import os
import sys
import io
import zipfile
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import duckdb
from typing import Any, Dict, List

# LangChain 및 Gemini 라이브러리
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain.callbacks import StdOutCallbackHandler
from langchain_core.callbacks.base import BaseCallbackHandler
import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix  
import matplotlib.pyplot as plt  
import seaborn as sns  


# =======================================================================
# 🛠️ 콜백 클래스 수정: BaseCallbackHandler 상속 (오류 해결)
# =======================================================================
class SimpleCollectCallback(BaseCallbackHandler):
    """
    LangChain AgentExecutor와의 호환성을 위해 BaseCallbackHandler를 상속받습니다.
    (원래의 CollectAllCallback 모듈 대체용)
    """
    def __init__(self):
        self.events = []
    
    def on_tool_error(self, error: Exception, **kwargs: Any) -> Any:
        # AgentExecutor가 오류 발생 시 호출하는 필수 메서드
        self.events.append({"type": "tool_error", "error": str(error)})
        
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        self.events.append({"type": "llm_start", "prompts": prompts})
        
    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        self.events.append({"type": "llm_end"})

    # AgentExecutor 호환성 유지를 위해 필수적인 최소 메서드 정의
    def on_agent_action(self, action: Any, **kwargs: Any) -> Any:
        pass
    
    def on_agent_finish(self, finish: Any, **kwargs: Any) -> Any:
        pass

# =============================
# 🚀 App 시작 및 설정
# =============================
load_dotenv()

st.set_page_config(page_title="DF Chatbot (Gemini)", page_icon="✨", layout="wide")
st.title("✨ DataFrame Chatbot (Gemini + LangChain)")
st.caption("Gemini + Python tool(ReAct)로 DataFrame을 분석하고 이상점을 검토합니다.")

# =============================
# 📁 CSV 로드 및 환경 변수
# =============================
# ⚠️ DATA_DIR 초기화
DEFAULT_DATA_DIR = os.getenv("DATA_DIR", "/Users/najongseong/dataset")
DFB_DEFAULT_NAME = "telemetry_raw.csv" # df_B 기본 파일명

# 세션 상태 초기화 및 DATA_DIR 설정
if "DATA_DIR" not in st.session_state:
    st.session_state["DATA_DIR"] = DEFAULT_DATA_DIR
if "df_A_data" not in st.session_state:
    st.session_state["df_A_data"] = None
if "df_A_name" not in st.session_state:
    st.session_state["df_A_name"] = "No Data"
if "csv_path" not in st.session_state:
    # 초기 csv_path는 기본 디렉토리의 기본 파일로 설정 (로드가 될 경우)
    st.session_state["csv_path"] = os.path.join(DEFAULT_DATA_DIR, "stormtrooper.csv") 

# 현재 사용 DATA_DIR
DATA_DIR = st.session_state["DATA_DIR"]
DFB_DEFAULT = os.path.join(DATA_DIR, DFB_DEFAULT_NAME)

# -----------------------------------------------------------------------
# 🔄 CSV 파일 목록 표시 및 선택 로직
# -----------------------------------------------------------------------

def load_df_A(path: str, display_name: str):
    """지정된 경로에서 df_A를 로드하고 세션 상태를 업데이트하는 헬퍼 함수"""
    try:
        new_df = pd.read_csv(path)
        st.session_state["df_A_data"] = new_df
        st.session_state["df_A_name"] = display_name
        # ✅ 선택한 파일의 전체 경로를 csv_path에 정확히 연결
        st.session_state["csv_path"] = path 
        return True, f"Loaded file: {display_name} (Shape: {new_df.shape})"
    except Exception as e:
        st.session_state["df_A_data"] = None
        st.session_state["df_A_name"] = "Load Failed"
        # 로드 실패 시 경로도 초기화
        st.session_state["csv_path"] = "" 
        return False, f"CSV 로드 실패: {path}\n{e}"

with st.sidebar:
    st.markdown("### 🗂️ 1. 데이터 폴더 설정")
    
    # DATA_DIR 입력 필드
    new_data_dir = st.text_input(
        "Enter CSV Directory Path",
        value=st.session_state["DATA_DIR"],
        key="data_dir_input"
    )

    if st.button("Set Directory"):
        if os.path.isdir(new_data_dir):
            st.session_state["DATA_DIR"] = new_data_dir
            st.session_state["df_A_data"] = None # 폴더 변경 시 데이터 초기화
            st.session_state["df_A_name"] = "No Data"
            # ✅ 폴더 변경 시 csv_path도 초기화하여 파일 목록을 새로 고치도록 유도
            st.session_state["csv_path"] = "" 
            st.success(f"Directory set to: `{new_data_dir}`")
            # 디렉토리 변경 후 재실행하여 파일 목록을 업데이트
            st.rerun() 
        else:
            st.error(f"Invalid directory path: `{new_data_dir}`")

    # 갱신된 DATA_DIR
    DATA_DIR = st.session_state["DATA_DIR"]
    DFB_DEFAULT = os.path.join(DATA_DIR, DFB_DEFAULT_NAME)

    st.markdown("---")
    st.markdown("### 📄 2. df_A CSV 파일 선택")
    st.caption(f"Search directory: `{DATA_DIR}`")
    
    # DATA_DIR에서 CSV 파일 목록 가져오기
    csv_files = []
    try:
        if os.path.isdir(DATA_DIR):
            for f in os.listdir(DATA_DIR):
                if f.lower().endswith('.csv'):
                    csv_files.append(f)
            csv_files.sort()
        else:
            st.warning("유효한 데이터 디렉토리를 설정해주세요.")
    except Exception as e:
        st.error(f"폴더 접근 오류: {e}")

    # 파일 선택 SelectBox
    selected_file = st.selectbox(
        "Select CSV file for df_A",
        options=["--- Select a file ---"] + csv_files,
        key="file_selector"
    )
    
    # 파일 로드 버튼
    if st.button("Load Selected File"):
        if selected_file and selected_file != "--- Select a file ---":
            file_path = os.path.join(DATA_DIR, selected_file)
            # 이 시점에서 load_df_A가 st.session_state["csv_path"]를 업데이트합니다.
            success, message = load_df_A(file_path, selected_file) 
            if success:
                st.success(message)
                st.rerun() # 파일 로드 후 UI 업데이트를 위해 재실행
            else:
                st.error(message)
        else:
            st.warning("파일을 선택해주세요.")

    st.markdown("---")
    # 현재 로드된 파일 경로 표시 (디버깅용)
    st.caption(f"**현재 로드 파일 경로:** `{st.session_state.get('csv_path', 'Not loaded')}`")
    st.caption(f"df_B 기본 가정 파일: `{os.path.basename(DFB_DEFAULT)}`")

# 최종 df 결정
df = st.session_state["df_A_data"]
CSV_PATH_DISPLAY = st.session_state["df_A_name"]

if df is None:
    st.error("분석할 DataFrame (df_A)을 로드하지 못했습니다. 유효한 디렉토리와 CSV 파일을 선택해주세요.")
    st.stop()
# -----------------------------------------------------------------------
# 이하 LLM 및 Agent 로직은 변경 없이 유지됩니다.
# -----------------------------------------------------------------------

# 환경 변수 확인
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY가 설정되어 있지 않습니다. .env 또는 환경변수를 확인하세요.")
    st.stop()

# =============================
# 🤖 LLM 세팅
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
# 🖼️ Preview
# =============================
st.subheader("Preview")
st.write(f"**Loaded CSV for df_A:** `{CSV_PATH_DISPLAY}` (Shape: {df.shape})")
st.dataframe(df.head(10), use_container_width=True)

# =============================
# 💬 Chat history
# =============================
history = StreamlitChatMessageHistory(key="lc_msgs:dfchat")

# =============================
# 🛠️ Tools 정의 및 Agent Globals (DATA_DIR 및 df 변수 업데이트)
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
    description=(
        "Execute Python on df_A/df_B with pandas/matplotlib. 'df' aliases df_A. "
        "Use this for custom computation, plotting (e.g., boxplot for outliers), and advanced data manipulation."
    )
)

# 2) 동적 CSV 로드 (loading_df)
@tool
def load_loading_csv(filename: str) -> str:
    """Load a CSV from DATA_DIR into 'loading_df' (for ad-hoc analysis). Pass only file name (e.g., 'loading_test.csv')."""
    # 툴 실행 시 현재 DATA_DIR을 사용
    current_data_dir = st.session_state.get("DATA_DIR", DEFAULT_DATA_DIR)
    path = os.path.join(current_data_dir, filename)
    try:
        new_df = pd.read_csv(path)
    except Exception as e:
        return f"Failed to load {path}: {e}"
    pytool.globals["loading_df"] = new_df
    preview = new_df.head(10).to_markdown(index=False)
    shape = f"{new_df.shape[0]} rows x {new_df.shape[1]} cols"
    return f"Loaded {filename} from {current_data_dir} into loading_df\nShape: {shape}\n\nPreview (head):\n{preview}"

# 3) 컬럼 요약 (loading_df 우선 → 없으면 df_A)
@tool
def describe_columns(cols: str = "") -> str:
    """
    Describe selected columns (comma-separated).
    Uses 'loading_df' if available; otherwise uses 'df_A'.
    This is useful for initial data quality checks and outlier boundary overview.
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

# 5) df_B 로더
@tool
def load_df_b(filename: str = "") -> str:
    """
    Load a CSV into 'df_B'. If 'filename' is empty, defaults to 'telemetry_raw.csv' under DATA_DIR.
    Example: load_df_b()
    """
    current_data_dir = st.session_state.get("DATA_DIR", DEFAULT_DATA_DIR)
    
    if not filename:
        path = os.path.join(current_data_dir, DFB_DEFAULT_NAME)
        show_name = os.path.basename(path)
    else:
        path = os.path.join(current_data_dir, filename) if not os.path.isabs(filename) else filename
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
    Run DuckDB SQL over df_A and (if loaded) df_B. Tables: df_A, df_B
    Example: SELECT a.key, a.metric FROM df_A a WHERE a.metric > 100 LIMIT 20
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
    load_df_b,
    describe_columns_on,
    sql_on_dfs,
    propose_join_keys,
    align_time_buckets,
]

# =============================
# 📜 Prompt (이상점 검토 강화)
# =============================
df_head_txt = df.head().to_string(index=False)



import os
# ... (생략된 import문) ...
from langchain_core.callbacks.base import BaseCallbackHandler
import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix  
import matplotlib.pyplot as plt  
import seaborn as sns  


# =======================================================================
# 🛠️ 콜백 클래스 수정: BaseCallbackHandler 상속 (오류 해결)
# ... (SimpleCollectCallback 클래스 정의 생략) ...
# =============================
# 🚀 App 시작 및 설정
# ... (App 시작 및 설정 코드 생략) ...
# =============================
# 📁 CSV 로드 및 환경 변수
# ... (CSV 로드 및 환경 변수 코드 생략) ...
# -----------------------------------------------------------------------
# 🔄 CSV 파일 목록 표시 및 선택 로직
# ... (CSV 로드 및 선택 로직 코드 생략) ...
# -----------------------------------------------------------------------
# 이하 LLM 및 Agent 로직은 변경 없이 유지됩니다.
# -----------------------------------------------------------------------

# 환경 변수 확인
# ... (환경 변수 확인 코드 생략) ...

# =============================
# 🤖 LLM 세팅
# ... (LLM 세팅 코드 생략) ...
# =============================
# 🖼️ Preview
# ... (Preview 코드 생략) ...
# =============================
# 💬 Chat history
# ... (Chat history 코드 생략) ...
# =============================
# 🛠️ Tools 정의 및 Agent Globals (DATA_DIR 및 df 변수 업데이트)
# ... (Tools 정의 코드 생략) ...
# 9) 타임버킷 정렬
# ... (align_time_buckets 도구 정의 생략) ...

tools = [
    pytool,
    load_loading_csv,
    describe_columns,
    save_plots_zip,
    load_df_b,
    describe_columns_on,
    sql_on_dfs,
    propose_join_keys,
    align_time_buckets,
]

# =============================
# 📜 Prompt (이상점 검토 강화)  <-- 🎯 이 섹션이 수정되었습니다.
# =============================
df_head_txt = df.head().to_string(index=False)

react_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are working with pandas dataframes. The main dataframe is `df_A` (alias: df). "
     "You may also load a related dataframe as `df_B` for root-cause analysis.\n\n"
     "Tools available:\n{tools}\nUse only tool names from: {tool_names}.\n"
     "If you use sql_on_dfs, available tables are df_A and (if loaded) df_B.\n\n"
     "Tool routing guide:\n"
     "- schema/summary/initial outlier bounds → describe_columns or describe_columns_on\n"
     "- load file → load_loading_csv or load_df_b\n"
     "- SQL/join/aggregation → sql_on_dfs\n"
     "- **custom compute/plots/outlier analysis** → **python_repl_ast**\n"
     "- suggest join keys → propose_join_keys\n"
     "- align timestamps to buckets → align_time_buckets\n\n"
     "**⚠️ 이상점(Outlier) 검토 지침:**\n"
     "1. **describe_columns**를 사용하여 수치형 컬럼의 min/max/std/사분위수(Q1, Q3)를 확인하세요.\n"
     "2. **python_repl_ast**를 사용하여 박스 플롯(Box Plot)을 그려 시각적으로 이상점을 확인하세요.\n"
     "   - 예시: `df['value_col'].plot(kind='box'); plt.title('Box Plot'); plt.show()`\n"
     "3. **python_repl_ast**를 사용하여 IQR(Q3 + 1.5*IQR) 등의 통계적 경계 조건을 계산하고, 이상점을 필터링하여 확인하세요.\n\n"
     # ✅ 변수 지속성 오류 방지를 위한 새 지침 추가
     "**🚨 Python 실행 필수 지침 (변수 지속성 오류 방지):**\n"
     "1. 복잡한 작업(데이터 준비, 모델 정의, 학습, 예측 등)은 변수 지속성을 보장하기 위해 **반드시 하나의 `python_repl_ast` 툴 호출 내**에서 실행하십시오.\n"
     "2. 코드를 여러 'Action:'으로 분할하지 마십시오. 변수(`model`, `X_train`, `results` 등)가 다음 단계에서 유실되어 오류가 발생합니다.\n"
     "3. 최종 결과를 사용자에게 보여줄 때는 코드 블록의 마지막에 `print(결과변수)` 명령을 포함하여 **Observation**으로 결과를 명확히 반환해야 합니다. 예를 들어, 예측 결과를 최종적으로 확인하는 경우 `print(results)` 또는 `print(accuracy)`를 사용하십시오.\n\n"
     # -------------------------------------------------------------
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
     f"This is the result of print(df_A.head()):\n{df_head_txt}\n"),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    ("assistant", "{agent_scratchpad}"),
])

# =============================
# ⚙️ ReAct Agent
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
# 🔄 RunnableWithMessageHistory
# =============================
agent_with_history = RunnableWithMessageHistory(
    agent,
    lambda session_id: history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# =============================
# 💻 UI 및 실행 로직
# =============================
st.write("---")
user_q = st.chat_input(
    "Ask about your data (예: 'telemetry_value 컬럼의 이상점을 박스 플롯으로 검토해줘', "
    "'describe_columns로 이상점의 경계값을 알려줘', "
    "'load_df_b() 후 df_A와 조인해서 분석해줘')"
)

if user_q:
    # 채팅 히스토리 기록
    history.add_user_message(user_q)

    left, right = st.columns([1, 1])
    with left:
        st.subheader("실시간 실행 로그")
        # StreamlitCallbackHandler는 중간 단계를 실시간으로 보여줍니다.
        st_cb = StreamlitCallbackHandler(st.container())

    # 사용자 정의 콜백 (타임라인 디버깅용)
    collector = SimpleCollectCallback()

    with st.spinner("Thinking with Gemini..."):
        try:
            result = agent_with_history.invoke(
                {"input": user_q},
                {
                    # StdOutCallbackHandler는 터미널에 로그를 출력합니다.
                    "callbacks": [st_cb, collector, StdOutCallbackHandler()],
                    "configurable": {"session_id": "konlo_ssid"},
                }
            )
        except Exception as e:
            st.error(f"Agent 실행 중 오류 발생: {e}")
            result = {"output": f"Agent 실행 중 오류 발생: {e}"}

    st.success("Done.")

    final = result.get("output", "Agent가 최종 답변을 생성하지 못했습니다.")
    with right:
        st.subheader("Answer")
        # LLM의 최종 답변을 채팅창에 표시
        history.add_ai_message(final)
        st.write(final)

        # 플롯 렌더링
        figs = [plt.figure(n) for n in plt.get_fignums()]
        for f in figs:
            f.set_size_inches(6, 4)
            f.set_dpi(100)
            st.pyplot(f, use_container_width=True)
        plt.close("all")

    # intermediate steps (디버깅 정보)
    st.write("---")
    st.subheader("intermediate_steps (툴 실행 상세)")
    steps = result.get("intermediate_steps", [])
    if steps:
        for i, (action, observation) in enumerate(steps, 1):
            with st.expander(f"Step {i}: {action.tool}"):
                st.markdown("**tool_input**")
                st.code(str(action.tool_input))
                st.markdown("**observation**")
                st.code(observation)
    else:
        st.info("intermediate_steps 비어 있음")

    # 콜백 타임라인 (디버깅 정보)
    st.write("---")
    st.subheader("콜백 이벤트 타임라인 (Simple Collect)")
    if collector.events:
        for j, e in enumerate(collector.events, 1):
            with st.expander(f"Event {j}: {e.get('type')}"):
                if e.get("type") == "llm_start":
                    prompts = e.get("prompts", [])
                    for k, p in enumerate(prompts, 1):
                        st.markdown(f"**Prompt {k}**")
                        st.code(p)
                else:
                    st.write(e)
    else:
        st.info("콜백 이벤트 정보 없음")

# =============================
# Streamlit 채팅 기록 표시
# =============================
st.sidebar.title("Chat History")
for msg in history.messages:
    if msg.type == "human":
        with st.sidebar:
            st.chat_message("user").write(msg.content)
    elif msg.type == "ai":
        # AI 메시지는 최종 답변이 이미 메인에 표시되었으므로, 여기서는 간단히 표시
        with st.sidebar:
            st.chat_message("assistant").write(msg.content[:50] + "...")