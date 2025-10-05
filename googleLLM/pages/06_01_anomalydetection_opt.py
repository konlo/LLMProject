import io
import os
import zipfile
from typing import Any, Dict, List, Tuple

import duckdb
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain.callbacks import StdOutCallbackHandler
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain_google_genai import ChatGoogleGenerativeAI


# =======================================================================
# Callback collection helper
# =======================================================================
class SimpleCollectCallback(BaseCallbackHandler):
    """Minimal callback handler to capture agent events for debugging."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def on_tool_error(self, error: Exception, **kwargs: Any) -> Any:
        self.events.append({"type": "tool_error", "error": str(error)})

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        self.events.append({"type": "llm_start", "prompts": prompts})

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        self.events.append({"type": "llm_end"})

    def on_agent_action(self, action: Any, **kwargs: Any) -> Any:  # pragma: no cover
        return None

    def on_agent_finish(self, finish: Any, **kwargs: Any) -> Any:  # pragma: no cover
        return None


# =======================================================================
# Streamlit configuration & defaults
# =======================================================================
load_dotenv()

PAGE_TITLE = "DF Chatbot (Gemini)"
PAGE_ICON = "✨"
PAGE_LAYOUT = "wide"
CHAT_INPUT_PLACEHOLDER = (
    "Ask about your data (예: 'telemetry_value 컬럼의 이상점을 박스 플롯으로 검토해줘', "
    "'describe_columns로 이상점의 경계값을 알려줘', "
    "'load_df_b() 후 df_A와 조인해서 분석해줘')"
)
DEFAULT_DATA_DIR = os.getenv("DATA_DIR", "/Users/najongseong/dataset")
DEFAULT_DF_B_NAME = "telemetry_raw.csv"
DEFAULT_PRIMARY_PLACEHOLDER = "stormtrooper.csv"
SELECT_PLACEHOLDER = "--- Select a file ---"
CHAT_HISTORY_KEY = "lc_msgs:dfchat"

SESSION_DEFAULTS = {
    "DATA_DIR": lambda: DEFAULT_DATA_DIR,
    "df_A_data": lambda: None,
    "df_A_name": lambda: "No Data",
    "csv_path": lambda: os.path.join(DEFAULT_DATA_DIR, DEFAULT_PRIMARY_PLACEHOLDER),
}


def configure_page() -> None:
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout=PAGE_LAYOUT)
    st.title("✨ DataFrame Chatbot (Gemini + LangChain)")
    st.caption("Gemini + Python tool(ReAct)로 DataFrame을 분석하고 이상점을 검토합니다.")


def init_session_state() -> None:
    for key, default in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default() if callable(default) else default


def get_active_data_dir() -> str:
    return st.session_state.get("DATA_DIR", DEFAULT_DATA_DIR)


def load_primary_dataframe(path: str, display_name: str) -> Tuple[bool, str]:
    """Load df_A from disk and update session state."""
    try:
        dataframe = pd.read_csv(path)
    except Exception as exc:  # pylint: disable=broad-except
        st.session_state["df_A_data"] = None
        st.session_state["df_A_name"] = "Load Failed"
        st.session_state["csv_path"] = ""
        return False, f"CSV 로드 실패: {path}\n{exc}"

    st.session_state["df_A_data"] = dataframe
    st.session_state["df_A_name"] = display_name
    st.session_state["csv_path"] = path
    return True, f"Loaded file: {display_name} (Shape: {dataframe.shape})"


def render_sidebar_controls() -> None:
    with st.sidebar:
        st.markdown("### 🗂️ 1. 데이터 폴더 설정")
        new_data_dir = st.text_input(
            "Enter CSV Directory Path",
            value=get_active_data_dir(),
            key="data_dir_input",
        )

        if st.button("Set Directory"):
            if os.path.isdir(new_data_dir):
                st.session_state["DATA_DIR"] = new_data_dir
                st.session_state["df_A_data"] = None
                st.session_state["df_A_name"] = "No Data"
                st.session_state["csv_path"] = ""
                st.success(f"Directory set to: `{new_data_dir}`")
                st.rerun()
            else:
                st.error(f"Invalid directory path: `{new_data_dir}`")

        current_dir = get_active_data_dir()
        st.markdown("---")
        st.markdown("### 📄 2. df_A CSV 파일 선택")
        st.caption(f"Search directory: `{current_dir}`")

        csv_files: List[str] = []
        try:
            if os.path.isdir(current_dir):
                csv_files = sorted(
                    f
                    for f in os.listdir(current_dir)
                    if f.lower().endswith(".csv")
                )
            else:
                st.warning("유효한 데이터 디렉토리를 설정해주세요.")
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"폴더 접근 오류: {exc}")

        selected_file = st.selectbox(
            "Select CSV file for df_A",
            options=[SELECT_PLACEHOLDER] + csv_files,
            key="file_selector",
        )

        if st.button("Load Selected File"):
            if selected_file and selected_file != SELECT_PLACEHOLDER:
                file_path = os.path.join(current_dir, selected_file)
                success, message = load_primary_dataframe(file_path, selected_file)
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
            else:
                st.warning("파일을 선택해주세요.")

        st.markdown("---")
        st.caption(
            f"**현재 로드 파일 경로:** `"
            f"{st.session_state.get('csv_path', 'Not loaded')}`"
        )
        st.caption(f"df_B 기본 가정 파일: `{DEFAULT_DF_B_NAME}`")


def ensure_dataframe_loaded() -> Tuple[pd.DataFrame, str]:
    dataframe = st.session_state.get("df_A_data")
    display_name = st.session_state.get("df_A_name", "No Data")
    if dataframe is None:
        st.error("분석할 DataFrame (df_A)을 로드하지 못했습니다. 유효한 디렉토리와 CSV 파일을 선택해주세요.")
        st.stop()
    return dataframe, display_name


def fetch_api_key_or_stop() -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("GOOGLE_API_KEY가 설정되어 있지 않습니다. .env 또는 환경변수를 확인하세요.")
        st.stop()
    return api_key


def create_llm(api_key: str) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=api_key,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        streaming=True,
    )


def render_preview(dataframe: pd.DataFrame, display_name: str) -> None:
    st.subheader("Preview")
    st.write(f"**Loaded CSV for df_A:** `{display_name}` (Shape: {dataframe.shape})")
    st.dataframe(dataframe.head(10), use_container_width=True)


def build_toolkit(dataframe: pd.DataFrame) -> List[Any]:
    pytool = PythonAstREPLTool(
        globals={
            "pd": pd,
            "plt": plt,
            "df": dataframe,
            "df_A": dataframe,
            "df_B": None,
            "loading_df": None,
        },
        name="python_repl_ast",
        description=(
            "Execute Python on df_A/df_B with pandas/matplotlib. 'df' aliases df_A. "
            "Use this for custom computation, plotting (e.g., boxplot for outliers), and advanced data manipulation."
        ),
    )

    @tool
    def load_loading_csv(filename: str) -> str:
        """Load a CSV from DATA_DIR into 'loading_df'."""
        current_dir = get_active_data_dir()
        path = os.path.join(current_dir, filename)
        try:
            new_df = pd.read_csv(path)
        except Exception as exc:  # pylint: disable=broad-except
            return f"Failed to load {path}: {exc}"
        pytool.globals["loading_df"] = new_df
        preview = new_df.head(10).to_markdown(index=False)
        shape = f"{new_df.shape[0]} rows x {new_df.shape[1]} cols"
        return (
            f"Loaded {filename} from {current_dir} into loading_df\n"
            f"Shape: {shape}\n\nPreview (head):\n{preview}"
        )

    @tool
    def describe_columns(cols: str = "") -> str:
        """Describe selected columns (loading_df preferred, fallback to df_A)."""
        if pytool.globals.get("loading_df") is not None:
            current_df = pytool.globals["loading_df"]
            source = "loading_df"
        else:
            current_df = pytool.globals.get("df_A") or dataframe
            source = "df_A"

        use_cols = [c.strip() for c in cols.split(",") if c.strip()] or list(current_df.columns)
        missing = [c for c in use_cols if c not in current_df.columns]
        if missing:
            return f"Missing columns: {missing}\nAvailable: {list(current_df.columns)}"

        desc = current_df[use_cols].describe(include="all").transpose()
        shape = f"{current_df.shape[0]} rows x {current_df.shape[1]} cols"
        return f"[source={source} | shape={shape}]\n\n" + desc.to_markdown()

    @tool
    def save_plots_zip() -> str:
        """Zip all current matplotlib figures. Use after plotting with python_repl_ast."""
        figures = [plt.figure(num) for num in plt.get_fignums()]
        if not figures:
            return "No figures to save."
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for index, figure in enumerate(figures, start=1):
                img = io.BytesIO()
                figure.savefig(img, format="png", dpi=100, bbox_inches="tight")
                img.seek(0)
                zf.writestr(f"plot_{index}.png", img.read())
        return f"Zipped {len(figures)} plots. Bytes={len(buffer.getvalue())}"

    @tool
    def load_df_b(filename: str = "") -> str:
        """Load a CSV into 'df_B'. Defaults to telemetry_raw.csv under DATA_DIR."""
        current_dir = get_active_data_dir()
        if not filename:
            path = os.path.join(current_dir, DEFAULT_DF_B_NAME)
            show_name = os.path.basename(path)
        else:
            path = os.path.join(current_dir, filename) if not os.path.isabs(filename) else filename
            show_name = os.path.basename(path)
        try:
            new_df = pd.read_csv(path)
        except Exception as exc:  # pylint: disable=broad-except
            return f"Failed to load df_B from {path}: {exc}"
        pytool.globals["df_B"] = new_df
        preview = new_df.head(10).to_markdown(index=False)
        return (
            f"Loaded df_B from '{show_name}' (full: {path}) with shape {new_df.shape}\n\n"
            f"Preview:\n{preview}"
        )

    @tool
    def describe_columns_on(target: str = "A", cols: str = "") -> str:
        """Describe columns from df_A or df_B."""
        target_clean = (target or "A").strip().upper()
        if target_clean == "B":
            df_b = pytool.globals.get("df_B")
            if df_b is None:
                return "df_B is not loaded. Use load_df_b() first."
            current_df = df_b
            source = "df_B"
        else:
            current_df = pytool.globals.get("df_A") or dataframe
            source = "df_A"

        use_cols = [c.strip() for c in cols.split(",") if c.strip()] or list(current_df.columns)
        missing = [c for c in use_cols if c not in current_df.columns]
        if missing:
            return f"Missing columns: {missing}\nAvailable: {list(current_df.columns)}"

        desc = current_df[use_cols].describe(include="all").transpose()
        shape = f"{current_df.shape[0]} rows x {current_df.shape[1]} cols"
        return f"[source={source} | shape={shape}]\n\n" + desc.to_markdown()

    @tool
    def sql_on_dfs(query: str) -> str:
        """Run DuckDB SQL over df_A and (if loaded) df_B."""
        try:
            duckdb.register("df_A", pytool.globals.get("df_A") or dataframe)
            if pytool.globals.get("df_B") is not None:
                duckdb.register("df_B", pytool.globals["df_B"])
            output = duckdb.sql(query).df()
            return output.head(200).to_markdown(index=False)
        except Exception as exc:  # pylint: disable=broad-except
            return f"SQL error: {exc}"

    @tool
    def propose_join_keys() -> str:
        """Suggest join key candidates between df_A and df_B."""
        df_a = pytool.globals.get("df_A") or dataframe
        df_b = pytool.globals.get("df_B")
        if df_b is None:
            return "df_B is not loaded."

        def dtype_signature(series: pd.Series) -> str:
            dtype_str = str(series.dtype)
            if "int" in dtype_str:
                return "int"
            if "float" in dtype_str:
                return "float"
            if any(keyword in dtype_str for keyword in ("datetime", "date", "time")):
                return "datetime"
            return "str"

        pairs = [
            (column, dtype_signature(df_a[column]))
            for column in df_a.columns
            if column in df_b.columns and dtype_signature(df_a[column]) == dtype_signature(df_b[column])
        ]

        if not pairs:
            return "No obvious same-name & same-type keys. Consider mapping tables or casting."

        markdown_rows = "\n".join([f"| {key} | {dtype_} |" for key, dtype_ in pairs])
        return "Candidate join keys (same name & type):\n| key | dtype |\n|---|---|\n" + markdown_rows

    @tool
    def align_time_buckets(target: str = "A", column: str = "ts", freq: str = "H") -> str:
        """Resample time-like column to buckets and store as df_A_bucketed or df_B_bucketed."""
        target_clean = (target or "A").strip().upper()
        if target_clean == "B":
            current_df = pytool.globals.get("df_B")
            if current_df is None:
                return "df_B is not loaded."
            df_key = "df_B_bucketed"
        else:
            current_df = pytool.globals.get("df_A") or dataframe
            df_key = "df_A_bucketed"

        if column not in current_df.columns:
            return f"Column '{column}' not in df_{target_clean}. Available: {list(current_df.columns)}"

        bucketed = current_df.copy()
        bucketed[column] = pd.to_datetime(bucketed[column], errors="coerce")
        bucket_col = f"{column}_bucket"
        bucketed[bucket_col] = bucketed[column].dt.to_period(freq).dt.to_timestamp()
        pytool.globals[df_key] = bucketed
        preview = bucketed[[bucket_col]].head().to_markdown(index=False)
        return f"Created {df_key} with '{bucket_col}' at freq={freq}.\nPreview:\n{preview}"

    return [
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


def build_react_prompt(dataframe: pd.DataFrame) -> ChatPromptTemplate:
    df_head_txt = dataframe.head().to_string(index=False)
    return ChatPromptTemplate.from_messages([
        (
            "system",
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
            "**🚨 Python 실행 필수 지침 (변수 지속성 오류 방지):**\n"
            "1. 복잡한 작업(데이터 준비, 모델 정의, 학습, 예측 등)은 변수 지속성을 보장하기 위해 **반드시 하나의 `python_repl_ast` 툴 호출 내**에서 실행하십시오.\n"
            "2. 코드를 여러 'Action:'으로 분할하지 마십시오. 변수(`model`, `X_train`, `results` 등)가 다음 단계에서 유실되어 오류가 발생합니다.\n"
            "3. 최종 결과를 사용자에게 보여줄 때는 코드 블록의 마지막에 `print(결과변수)` 명령을 포함하여 **Observation**으로 결과를 명확히 반환해야 합니다. 예를 들어, 예측 결과를 최종적으로 확인하는 경우 `print(results)` 또는 `print(accuracy)`를 사용하십시오.\n\n"
            "Parsing recovery:\n"
            "- If you output 'Action:' without 'Action Input:', immediately continue with only 'Action Input: <...>'.\n"
            "- Do not wrap code in backticks.\n"
            "- Keep Action Input minimal, valid, and executable.\n\n"
            """
            "   **실행 규칙:**\n"
            "   1. 각 도구는 최대 3번까지만 시도\n"
            "   2. 동일한 오류가 2번 발생하면 즉시 다른 방법 시도\n"
            "   3. 무한 루프 감지 시 자동으로 중단\n\n"
            "   **오류별 대처법:**\n"
            "   - TypeError (sort_values): → np.argsort() 또는 sorted() 사용\n"
            "   - KeyError: → 컬럼명 확인 후 재시도\n"
            "   - AttributeError: → 객체 타입 확인 후 적절한 메서드 사용\n\n"
            "   **상태 체크포인트:**\n"
            "   매 3번째 액션마다:\n"
            "   - 진행 상황 요약\n"
            "   - 목표 달성도 확인\n"
            "   - 필요시 전략 수정\n"
            """
            "ALWAYS follow this exact format:\n"
            "Question: <restated question>\n"
            "Thought: <brief reasoning>\n"
            "Action: <ONE tool name from {tool_names}>\n"
            "Action Input: <valid code with NO backticks>\n"
            "Observation: <tool result>\n"
            "(Repeat Thought/Action/Action Input/Observation as needed)\n"
            "Thought: I now know the final answer\n"
            "Final Answer: <answer>\n\n"
            f"This is the result of print(df_A.head()):\n{df_head_txt}\n",
        ),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        ("assistant", "{agent_scratchpad}"),
    ])


def create_agent_executor(
    llm: ChatGoogleGenerativeAI, tools: List[Any], prompt: ChatPromptTemplate
) -> AgentExecutor:
    react_runnable = create_react_agent(llm, tools, prompt=prompt)
    return AgentExecutor(
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


def build_agent_with_history(agent: AgentExecutor, history: StreamlitChatMessageHistory) -> RunnableWithMessageHistory:
    return RunnableWithMessageHistory(
        agent,
        lambda session_id: history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )


def handle_user_query(agent_with_history: RunnableWithMessageHistory) -> None:
    st.write("---")
    user_question = st.chat_input(CHAT_INPUT_PLACEHOLDER)
    if not user_question:
        return

    left, right = st.columns([1, 1])
    with left:
        st.subheader("실시간 실행 로그")
        streamlit_callback = StreamlitCallbackHandler(st.container())

    collector = SimpleCollectCallback()

    with st.spinner("Thinking with Gemini..."):
        try:
            result = agent_with_history.invoke(
                {"input": user_question},
                {
                    "callbacks": [streamlit_callback, collector, StdOutCallbackHandler()],
                    "configurable": {"session_id": "konlo_ssid"},
                },
            )
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Agent 실행 중 오류 발생: {exc}")
            result = {"output": f"Agent 실행 중 오류 발생: {exc}"}

    st.success("Done.")

    final_output = result.get("output", "Agent가 최종 답변을 생성하지 못했습니다.")
    with right:
        st.subheader("Answer")
        st.write(final_output)

        figures = [plt.figure(num) for num in plt.get_fignums()]
        for figure in figures:
            figure.set_size_inches(6, 4)
            figure.set_dpi(100)
            st.pyplot(figure, use_container_width=True)
        plt.close("all")

    st.write("---")
    st.subheader("intermediate_steps (툴 실행 상세)")
    steps = result.get("intermediate_steps", [])
    if steps:
        for index, (action, observation) in enumerate(steps, start=1):
            with st.expander(f"Step {index}: {action.tool}"):
                st.markdown("**tool_input**")
                st.code(str(action.tool_input))
                st.markdown("**observation**")
                st.code(observation)
    else:
        st.info("intermediate_steps 비어 있음")

    st.write("---")
    st.subheader("콜백 이벤트 타임라인 (Simple Collect)")
    if collector.events:
        for index, event in enumerate(collector.events, start=1):
            with st.expander(f"Event {index}: {event.get('type')}"):
                if event.get("type") == "llm_start":
                    for prompt_index, prompt in enumerate(event.get("prompts", []), start=1):
                        st.markdown(f"**Prompt {prompt_index}**")
                        st.code(prompt)
                else:
                    st.write(event)
    else:
        st.info("콜백 이벤트 정보 없음")


def render_history_sidebar(history: StreamlitChatMessageHistory) -> None:
    st.sidebar.title("Chat History")
    for message in history.messages:
        if message.type == "human":
            st.sidebar.chat_message("user").write(message.content)
        elif message.type == "ai":
            st.sidebar.chat_message("assistant").write(message.content[:50] + "...")


def main() -> None:
    configure_page()
    init_session_state()
    render_sidebar_controls()

    dataframe, display_name = ensure_dataframe_loaded()
    api_key = fetch_api_key_or_stop()
    llm = create_llm(api_key)

    render_preview(dataframe, display_name)

    history = StreamlitChatMessageHistory(key=CHAT_HISTORY_KEY)
    tools = build_toolkit(dataframe)
    prompt = build_react_prompt(dataframe)
    agent = create_agent_executor(llm, tools, prompt)
    agent_with_history = build_agent_with_history(agent, history)

    handle_user_query(agent_with_history)
    render_history_sidebar(history)


main()
