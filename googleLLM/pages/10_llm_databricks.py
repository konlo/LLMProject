"""Streamlit app for Databricks-backed LLM EDA.

Loads a Databricks table into a DataFrame, then exposes the familiar
outlier-focused toolset powered by Gemini."""

from typing import Any, Dict, List, Optional
import os

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from databricks import sql as dbsql
from dotenv import load_dotenv

from langchain.agents import AgentExecutor, create_react_agent
from langchain.callbacks import StdOutCallbackHandler
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import render_text_description, tool
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    from sklearn.ensemble import IsolationForest
except Exception:  # pragma: no cover - optional dependency
    IsolationForest = None

try:
    from statsmodels.tsa.seasonal import STL
except Exception:  # pragma: no cover - optional dependency
    STL = None


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
load_dotenv()

st.set_page_config(page_title="DF Chatbot (Gemini) — Databricks", page_icon="✨", layout="wide")
st.title("✨ DataFrame Chatbot (Gemini + Databricks)")
st.caption("Databricks table loader → EDA + LLM Agent")

DB_HOST = os.getenv("DATABRICKS_HOST")
DB_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH")
DB_TOKEN = os.getenv("DATABRICKS_TOKEN")
TABLE_FQN = os.getenv("TABLE_FQN", "default.stormtrooper")
DEFAULT_LIMIT = int(os.getenv("DEFAULT_LIMIT", "20000"))

TIME_COLUMN_CANDIDATES = [
    "datetime",
    "timestamp",
    "ts",
    "time",
    "event_time",
    "eventtime",
    "date",
    "created_at",
]

pytool: Optional[PythonAstREPLTool] = None
TOOLS: List[Any] = []


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_df_from_databricks(query: str) -> pd.DataFrame:
    """Execute SQL on Databricks SQL Warehouse and return a DataFrame."""
    with dbsql.connect(server_hostname=DB_HOST, http_path=DB_HTTP_PATH, access_token=DB_TOKEN) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
    return pd.DataFrame(rows, columns=cols)


@st.cache_data(show_spinner=False)
def load_table(table_fqn: str, limit: int = DEFAULT_LIMIT) -> pd.DataFrame:
    query = f"SELECT * FROM {table_fqn} LIMIT {int(limit)}"
    return fetch_df_from_databricks(query)


def _parse_int(val: Any, default: int) -> int:
    if val is None:
        return default
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, float):
        try:
            return int(val)
        except Exception:
            return default
    if isinstance(val, str):
        stripped = val.strip()
        if stripped == "":
            return default
        try:
            return int(float(stripped))
        except Exception:
            return default
    return default


def _parse_float(val: Any, default: float) -> float:
    if val is None:
        return default
    if isinstance(val, (int, float, np.integer, np.floating)):
        try:
            return float(val)
        except Exception:
            return default
    if isinstance(val, str):
        stripped = val.strip()
        if stripped == "":
            return default
        try:
            return float(stripped)
        except Exception:
            return default
    return default


def _resolve_time_column(df: Optional[pd.DataFrame], preferred: str) -> Optional[str]:
    if df is None or not isinstance(df, pd.DataFrame) or not preferred:
        return None
    if preferred in df.columns:
        return preferred
    lower_map = {col.lower(): col for col in df.columns}
    pref_lower = preferred.lower()
    if pref_lower in lower_map:
        return lower_map[pref_lower]
    for alias in TIME_COLUMN_CANDIDATES:
        if alias == preferred:
            continue
        if alias in df.columns:
            return alias
        alias_lower = alias.lower()
        if alias_lower in lower_map:
            return lower_map[alias_lower]
    return None


class SimpleCollectCallback(BaseCallbackHandler):
    """Collects callback events for inspection."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def on_tool_error(self, error: Exception, **kwargs: Any) -> None:
        self.events.append({"type": "tool_error", "error": str(error)})

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        self.events.append({"type": "llm_start", "prompts": prompts})

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        self.events.append({"type": "llm_end"})


def init_session_state() -> None:
    defaults = {
        "df_A_data": None,
        "df_A_name": TABLE_FQN,
        "df_B_data": None,
        "df_B_name": "Not loaded",
        "explanation_lang": "English",
        "row_limit": DEFAULT_LIMIT,
        "where_clause": "",
        "order_by": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar(table_fqn: str) -> None:
    with st.sidebar:
        st.markdown("### 💬 EDA 설명 언어")
        st.session_state["explanation_lang"] = st.selectbox(
            "Agent 요약 언어",
            options=["English", "한국어"],
            index=0 if st.session_state.get("explanation_lang", "English") == "English" else 1,
        )

        st.markdown("### 🗃️ Databricks Table Loader")
        st.caption(f"Reading from **{table_fqn}**")
        st.session_state["row_limit"] = st.number_input(
            "LIMIT",
            min_value=1000,
            max_value=2_000_000,
            step=1000,
            value=st.session_state["row_limit"],
        )
        st.session_state["where_clause"] = st.text_input(
            "WHERE (optional)",
            value=st.session_state["where_clause"],
            placeholder="e.g., model='PM9A3' AND datetime >= '2025-01-01'",
        )
        st.session_state["order_by"] = st.text_input(
            "ORDER BY (optional)",
            value=st.session_state["order_by"],
            placeholder="e.g., datetime DESC",
        )

        if st.button("Load from Databricks"):
            query = f"SELECT * FROM {table_fqn}"
            where_clause = st.session_state["where_clause"].strip()
            order_by = st.session_state["order_by"].strip()
            limit_value = int(st.session_state["row_limit"])

            if where_clause:
                query += f" WHERE {where_clause}"
            if order_by:
                query += f" ORDER BY {order_by}"
            query += f" LIMIT {limit_value}"

            try:
                df_loaded = fetch_df_from_databricks(query)
                st.session_state["df_A_data"] = df_loaded
                st.session_state["df_A_name"] = f"{table_fqn} (LIMIT {limit_value})"
                st.success(f"Loaded {len(df_loaded):,} rows from Databricks.")
            except Exception as exc:
                st.error(f"Databricks load failed: {exc}")


def auto_load_initial_sample(table_fqn: str, databricks_ready: bool) -> None:
    if st.session_state["df_A_data"] is None and databricks_ready:
        try:
            st.info("Loading initial sample from Databricks…")
            df_loaded = load_table(table_fqn, DEFAULT_LIMIT)
            st.session_state["df_A_data"] = df_loaded
            st.session_state["df_A_name"] = f"{table_fqn} (LIMIT {DEFAULT_LIMIT})"
            st.success("Initial load complete.")
        except Exception as exc:
            st.error(f"Initial load failed: {exc}")


def ensure_primary_dataframe() -> pd.DataFrame:
    df_A = st.session_state.get("df_A_data")
    if df_A is None or len(df_A) == 0:
        st.error("Databricks에서 데이터를 불러오지 못했습니다. 환경변수/권한/HTTP Path를 확인하세요.")
        st.stop()
    return df_A


def render_preview(df_A: pd.DataFrame) -> None:
    st.subheader("Preview")
    st.write(f"**Loaded table:** `{st.session_state['df_A_name']}` (Shape: {df_A.shape})")
    st.dataframe(df_A.head(10), use_container_width=True)


# -----------------------------------------------------------------------------
# Tool definitions (depend on pytool globals)
# -----------------------------------------------------------------------------
@tool
def describe_columns(cols: str = "") -> str:
    """Describe selected columns (comma-separated) from df_A."""
    if pytool is None:
        return "pytool not initialised."
    current_df = pytool.globals.get("df_A")
    if current_df is None:
        return "df_A not loaded."

    column_list = [col.strip() for col in (cols or "").split(",") if col.strip()]
    use_cols = column_list or list(current_df.columns)
    missing = [col for col in use_cols if col not in current_df.columns]
    if missing:
        return f"Missing columns: {missing}\nAvailable: {list(current_df.columns)}"

    summary = current_df[use_cols].describe(include="all").transpose()
    shape = f"{current_df.shape[0]} rows x {current_df.shape[1]} cols"
    return f"[source=df_A | shape={shape}]\n\n" + summary.to_markdown()


@tool
def sql_on_dfs(query: str) -> str:
    """Run DuckDB SQL over df_A, df_B (if loaded), df_join (if created)."""
    if pytool is None:
        return "pytool not initialised."
    try:
        if pytool.globals.get("df_A") is not None:
            duckdb.register("df_A", pytool.globals["df_A"])
        if pytool.globals.get("df_B") is not None:
            duckdb.register("df_B", pytool.globals["df_B"])
        if pytool.globals.get("df_join") is not None:
            duckdb.register("df_join", pytool.globals["df_join"])
        result = duckdb.sql(query).df()
        return result.head(200).to_markdown(index=False)
    except Exception as exc:
        return f"SQL error: {exc}"


@tool
def select_numeric_candidates(min_unique: Any = 10, min_std: Any = 1e-9) -> str:
    """Return numeric columns in df_A that meet uniqueness/variance thresholds."""
    if pytool is None:
        return "pytool not initialised."
    df_A = pytool.globals.get("df_A")
    if df_A is None:
        return "df_A not loaded."

    min_unique_val = _parse_int(min_unique, 10)
    min_std_val = _parse_float(min_std, 1e-9)
    numeric_cols: List[str] = []

    for col in df_A.columns:
        if pd.api.types.is_numeric_dtype(df_A[col]):
            unique_count = df_A[col].nunique(dropna=True)
            std_val = pd.to_numeric(df_A[col], errors="coerce").std(skipna=True)
            if unique_count >= min_unique_val and std_val is not None and std_val > min_std_val and np.isfinite(std_val):
                numeric_cols.append(col)

    if not numeric_cols:
        return "No numeric candidates."
    return "Numeric candidates:\n" + pd.DataFrame({"column": numeric_cols}).to_markdown(index=False)


@tool
def rank_outlier_columns(method: str = "iqr_ratio", top_n: Any = 20) -> str:
    """Rank df_A numeric columns by IQR-based outlier rate."""
    if pytool is None:
        return "pytool not initialised."
    df_A = pytool.globals.get("df_A")
    if df_A is None:
        return "df_A not loaded."

    top_n_val = _parse_int(top_n, 20)
    rows: List[Dict[str, float]] = []

    for col in df_A.columns:
        if pd.api.types.is_numeric_dtype(df_A[col]) and df_A[col].nunique(dropna=True) >= 10:
            series = pd.to_numeric(df_A[col], errors="coerce")
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            if not np.isfinite(iqr) or iqr == 0:
                continue
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            rate = float(((series < lower) | (series > upper)).mean() * 100.0)
            rows.append({"column": col, "outlier_rate_%": rate, "lo": float(lower), "hi": float(upper)})

    if not rows:
        return "No IQR-detectable outliers."

    rank_df = pd.DataFrame(rows).sort_values("outlier_rate_%", ascending=False).head(top_n_val)
    pytool.globals["df_outlier_rank"] = rank_df
    return rank_df.to_markdown(index=False)


@tool
def anomaly_iqr(col: str) -> str:
    """Flag IQR-based outliers for the requested df_A column."""
    if pytool is None:
        return "pytool not initialised."
    df_A = pytool.globals.get("df_A")
    if df_A is None:
        return "df_A not loaded."

    column_name = str(col).strip()
    if column_name not in df_A.columns:
        return f"Column '{column_name}' not found."

    series = pd.to_numeric(df_A[column_name], errors="coerce")
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    flags = (series < lower) | (series > upper)
    df_A[f"{column_name}_is_outlier_iqr"] = flags
    pytool.globals["df_A"] = df_A
    return f"[anomaly_iqr] col='{column_name}', bounds=({lower:.3f},{upper:.3f}), outliers={int(flags.sum())}"


@tool
def anomaly_isoforest(cols: str, contamination: Any = 0.01, random_state: Any = 42) -> str:
    """Run IsolationForest on selected df_A columns and mark outliers."""
    if IsolationForest is None:
        return "scikit-learn이 설치되어 있지 않습니다. (pip install scikit-learn)"
    if pytool is None:
        return "pytool not initialised."
    df_A = pytool.globals.get("df_A")
    if df_A is None:
        return "df_A not loaded."

    contamination_val = _parse_float(contamination, 0.01)
    random_state_val = _parse_int(random_state, 42)
    targets = [col.strip() for col in str(cols).split(",") if col.strip()]

    for col in targets:
        if col not in df_A.columns:
            return f"Column '{col}' not found."

    X = df_A[targets].apply(pd.to_numeric, errors="coerce").dropna()
    if X.shape[0] < 20:
        return "Not enough rows for IsolationForest (>=20 recommended)."

    clf = IsolationForest(n_estimators=200, contamination=contamination_val, random_state=random_state_val)
    predictions = clf.fit_predict(X.values)  # -1 == outlier
    flags = pd.Series(predictions == -1, index=X.index)

    df_A["isoforest_outlier"] = False
    df_A.loc[flags.index, "isoforest_outlier"] = flags
    pytool.globals["df_A"] = df_A
    return f"[anomaly_isoforest] cols={targets}, contamination={contamination_val}, outliers={int(flags.sum())}"


@tool
def auto_outlier_eda(top_n: Any = 10, on: str = "datetime") -> str:
    """Perform the automated outlier EDA pipeline over df_A."""
    if pytool is None:
        return "pytool not initialised."
    df_A = pytool.globals.get("df_A")
    if df_A is None:
        return "df_A not loaded."

    top_n_val = _parse_int(top_n, 10)
    numeric_cols = [
        col
        for col in df_A.columns
        if pd.api.types.is_numeric_dtype(df_A[col]) and df_A[col].nunique(dropna=True) >= 10
    ]
    if not numeric_cols:
        return "No numeric candidates."

    rows: List[Dict[str, float]] = []
    for col in numeric_cols:
        series = pd.to_numeric(df_A[col], errors="coerce")
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        rate = float(((series < lower) | (series > upper)).mean() * 100.0)
        rows.append({"column": col, "outlier_rate_%": rate, "lo": float(lower), "hi": float(upper)})

    if not rows:
        return "No IQR-detectable outliers."

    rank_df = pd.DataFrame(rows).sort_values("outlier_rate_%", ascending=False)
    top_cols = rank_df.head(max(1, min(top_n_val, len(rank_df))))["column"].tolist()
    pytool.globals["df_outlier_rank"] = rank_df

    summary: List[str] = [
        "[auto_outlier_eda] IQR scan complete.",
        "**Top outlier columns:**\n" + rank_df.head(20).to_markdown(index=False),
    ]

    time_col = _resolve_time_column(df_A, on)
    if time_col and top_cols and STL is not None:
        try:
            focus_col = top_cols[0]
            ts = pd.to_datetime(df_A[time_col], errors="coerce")
            values = pd.to_numeric(df_A[focus_col], errors="coerce")
            ok = ts.notna() & values.notna()
            ts, values = ts[ok], values[ok]
            series = pd.Series(values.values, index=ts).sort_index()
            if len(series) >= 48:
                res = STL(series, period=24, robust=True).fit()
                resid = pd.Series(res.resid, index=series.index)
                max_abs = float(resid.abs().max())
                when = str(resid.abs().idxmax())
                summary.append(f"**STL residual spike** for '{focus_col}': max |resid|={max_abs:.3f} at {when}")
        except Exception:
            pass

    if IsolationForest is not None and len(top_cols) >= 2:
        try:
            subset_cols = top_cols[: min(4, len(top_cols))]
            matrix = df_A[subset_cols].apply(pd.to_numeric, errors="coerce").dropna()
            if len(matrix) >= 50:
                clf = IsolationForest(n_estimators=200, contamination=0.02, random_state=42).fit(matrix.values)
                outlier_rate = float((clf.predict(matrix.values) == -1).mean() * 100.0)
                summary.append(f"**IsolationForest** on {subset_cols} → outlier_rate≈{outlier_rate:.1f}%")
        except Exception:
            pass

    return "\n\n".join(summary)


def setup_tools(df_A: pd.DataFrame, df_B: Optional[pd.DataFrame]) -> List[Any]:
    global pytool, TOOLS
    pytool = PythonAstREPLTool(
        globals={
            "pd": pd,
            "np": np,
            "plt": plt,
            "df": df_A,
            "df_A": df_A,
            "df_B": df_B,
            "df_join": None,
            "duckdb": duckdb,
        },
        name="python_repl_ast",
        description="Execute Python on df_A/df_B/df_join with pandas/matplotlib.",
    )
    TOOLS = [
        pytool,
        describe_columns,
        sql_on_dfs,
        select_numeric_candidates,
        rank_outlier_columns,
        anomaly_iqr,
        anomaly_isoforest,
        auto_outlier_eda,
    ]
    return TOOLS


# -----------------------------------------------------------------------------
# Agent & UI
# -----------------------------------------------------------------------------
def build_agent(df_A: pd.DataFrame, df_B: Optional[pd.DataFrame], tools: List[Any]) -> tuple:
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        st.error("GOOGLE_API_KEY가 설정되어 있지 않습니다. .env 또는 환경변수를 확인하세요.")
        st.stop()

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        api_key=google_api_key,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        disable_streaming=False,
    )

    headA = df_A.head().to_string(index=False)
    headB = df_B.head().to_string(index=False) if df_B is not None else "(df_B not loaded)"

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert data analyst for SSD telemetry and tabular data. "
                "You work with two dataframes: df_A (main, alias: df) and df_B (optional).\n\n"
                "When the user asks for outlier-focused EDA, DO NOT ask questions. Immediately run this pipeline:\n"
                "  1) describe_columns → select_numeric_candidates → rank_outlier_columns\n"
                "  2) anomaly_iqr on top-N columns\n"
                "  3) stl decomposition best-effort if time series available\n"
                "  4) isolation forest on k-best if sklearn available\n"
                "  5) Summarize: top outlier columns + time spikes + next steps.\n\n"
                "ALWAYS follow this EXACT format:\n"
                "Question: <restated question>\n"
                "Thought: <brief reasoning>\n"
                "Action: <ONE tool name from {tool_names}>\n"
                "Action Input: <valid input with NO backticks>\n"
                "Observation: <tool result>\n"
                "(Repeat Thought/Action/Action Input/Observation as needed)\n"
                "Thought: I now know the final answer\n"
                "Final Answer: <concise answer>\n\n"
                "If you output anything outside this format, continue immediately by outputting ONLY a valid 'Action' and 'Action Input'.\n\n"
                f"df_A.head():\n{headA}\n\n"
                f"df_B.head():\n{headB}\n",
            ),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            ("assistant", "{agent_scratchpad}"),
        ]
    )

    tool_desc = render_text_description(tools)
    tool_names = ", ".join(tool.name for tool in tools)
    prompt = prompt.partial(tools=tool_desc, tool_names=tool_names)

    react_runnable = create_react_agent(llm, tools, prompt=prompt)
    agent = AgentExecutor(
        agent=react_runnable,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        max_iterations=20,
        handle_parsing_errors=(
            "PARSING ERROR. DO NOT APOLOGIZE. Immediately continue by outputting ONLY:\n"
            "Action: describe_columns\n"
            "Action Input: \n"
        ),
    )

    history = StreamlitChatMessageHistory(key="lc_msgs:dfchat")
    agent_with_history = RunnableWithMessageHistory(
        agent,
        lambda session_id: history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    return agent_with_history, llm, history


def render_conversational_ui(agent_with_history: RunnableWithMessageHistory, llm: ChatGoogleGenerativeAI) -> None:
    st.write("---")
    user_query = st.chat_input(
        "예) 이상점 EDA 해줘 / auto_outlier_eda() / rank_outlier_columns(top_n=15) / "
        "anomaly_iqr('temperature') / sql_on_dfs('select count(*) from df_A')"
    )

    if not user_query:
        return

    left, right = st.columns([1, 1])
    with left:
        st.subheader("실시간 실행 로그")
        streamlit_cb = StreamlitCallbackHandler(st.container())

    collector = SimpleCollectCallback()

    with st.spinner("Thinking with Gemini..."):
        try:
            result = agent_with_history.invoke(
                {"input": user_query},
                {
                    "callbacks": [streamlit_cb, collector, StdOutCallbackHandler()],
                    "configurable": {"session_id": "db_loader_eda"},
                },
            )
        except Exception as exc:
            st.error(f"Agent 실행 중 오류: {exc}")
            result = {"output": f"Agent 실행 중 오류: {exc}"}

    st.success("Done.")
    final_text = result.get("output", "Agent가 최종 답변을 생성하지 못했습니다.")
    final_text = final_text if isinstance(final_text, str) else str(final_text)

    with right:
        st.subheader("Answer")
        lang_choice = st.session_state.get("explanation_lang", "English")
        display_answer = final_text

        if lang_choice == "한국어" and final_text.strip():
            try:
                translation_prompt = "다음 분석 결과를 자연스럽고 간결한 한국어로 설명해줘.\n\n" + final_text
                translated_msg = llm.invoke(translation_prompt)
                translated_text = getattr(translated_msg, "content", None)
                if translated_text:
                    display_answer = translated_text
            except Exception as exc:
                st.warning(f"한국어 번역 중 오류: {exc}")

        st.write(display_answer)

        st.markdown("---")
        st.subheader("EDA Visualizations")

        visuals_rendered = False
        if pytool is not None:
            outlier_rank_df = pytool.globals.get("df_outlier_rank")
            if isinstance(outlier_rank_df, pd.DataFrame) and not outlier_rank_df.empty:
                if {"column", "outlier_rate_%"} <= set(outlier_rank_df.columns):
                    visuals_rendered = True
                    top_outliers = outlier_rank_df.head(15).set_index("column")["outlier_rate_%"]
                    st.markdown("**Top Outlier Columns (IQR %)**")
                    st.bar_chart(top_outliers)

            current_df_a = pytool.globals.get("df_A")
            if isinstance(current_df_a, pd.DataFrame) and not current_df_a.empty:
                outlier_cols = [col for col in current_df_a.columns if col.endswith("_is_outlier_iqr")]
                if "isoforest_outlier" in current_df_a.columns or outlier_cols:
                    visuals_rendered = True
                    st.markdown("**Outlier Flags Overview**")
                    with st.container():
                        if "isoforest_outlier" in current_df_a.columns:
                            iso_counts = current_df_a["isoforest_outlier"].value_counts(dropna=False)
                            st.write(
                                {
                                    "isoforest_outlier=True": int(iso_counts.get(True, 0)),
                                    "isoforest_outlier=False": int(iso_counts.get(False, 0)),
                                }
                            )
                        if outlier_cols:
                            outlier_summary = (
                                current_df_a[outlier_cols]
                                .apply(lambda series: int(series.fillna(False).sum()))
                                .rename("outlier_count")
                                .to_frame()
                            )
                            st.table(outlier_summary)

        figures = [plt.figure(num) for num in plt.get_fignums()]
        for fig in figures:
            fig.set_size_inches(6, 4)
            fig.set_dpi(100)
            st.pyplot(fig, use_container_width=True)
            visuals_rendered = True
        plt.close("all")

        if not visuals_rendered:
            st.info("시각화 결과가 없습니다. 필요한 경우 툴을 실행해보세요.")

    st.write("---")
    st.subheader("intermediate_steps (툴 실행 상세)")
    steps = result.get("intermediate_steps", [])
    if steps:
        for idx, (action, observation) in enumerate(steps, start=1):
            with st.expander(f"Step {idx}: {action.tool}"):
                st.markdown("**tool_input**")
                st.code(str(action.tool_input))
                st.markdown("**observation**")
                observation_text = observation if isinstance(observation, str) else str(observation)
                st.code(observation_text[:4000])
    else:
        st.info("intermediate_steps 비어 있음")


def render_chat_history(history: StreamlitChatMessageHistory) -> None:
    st.sidebar.title("Chat History")
    for msg in history.messages:
        if msg.type == "human":
            with st.sidebar:
                st.chat_message("user").write(msg.content)
        elif msg.type == "ai":
            with st.sidebar:
                st.chat_message("assistant").write(msg.content[:50] + "...")


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------
def main() -> None:
    init_session_state()

    databricks_ready = bool(DB_HOST and DB_HTTP_PATH and DB_TOKEN)
    print(databricks_ready)
    if not databricks_ready:
        st.error("DATABRICKS_HOST / DATABRICKS_HTTP_PATH / DATABRICKS_TOKEN 환경변수가 필요합니다.")

    render_sidebar(TABLE_FQN)
    auto_load_initial_sample(TABLE_FQN, databricks_ready)

    df_A = ensure_primary_dataframe()
    df_B = st.session_state.get("df_B_data")

    render_preview(df_A)

    tools = setup_tools(df_A, df_B)
    agent_with_history, llm, history = build_agent(df_A, df_B, tools)

    render_conversational_ui(agent_with_history, llm)
    render_chat_history(history)


if __name__ == "__main__":
    main()
