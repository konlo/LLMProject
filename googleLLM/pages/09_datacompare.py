import os
import sys
import io
import zipfile
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import duckdb

# Optional deps
try:
    from sklearn.ensemble import IsolationForest
except Exception:
    IsolationForest = None

try:
    from statsmodels.tsa.seasonal import STL
except Exception:
    STL = None

# LangChain & Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain.callbacks import StdOutCallbackHandler
from langchain_core.tools import render_text_description  # for {tools}/{tool_names}

# =======================================================================
# ✅ 공용 파서 유틸 (빈 문자열/None/문자열 숫자도 안전 변환)
# =======================================================================
def _parse_int(val, default: int) -> int:
    """빈 문자열/None/문자열/정수 모두 안전 변환."""
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
        s = val.strip()
        if s == "":
            return default
        try:
            return int(float(s))
        except Exception:
            return default
    return default

def _parse_float(val, default: float) -> float:
    """빈 문자열/None/문자열/실수 모두 안전 변환."""
    if val is None:
        return default
    if isinstance(val, (int, float, np.integer, np.floating)):
        try:
            return float(val)
        except Exception:
            return default
    if isinstance(val, str):
        s = val.strip()
        if s == "":
            return default
        try:
            return float(s)
        except Exception:
            return default
    return default

# =======================================================================
# 🛠️ 콜백 수집기
# =======================================================================
class SimpleCollectCallback(BaseCallbackHandler):
    """
    LangChain AgentExecutor와의 호환성을 위해 BaseCallbackHandler 상속.
    중간 이벤트/에러를 수집하여 UI에 노출.
    """
    def __init__(self):
        self.events = []
    
    def on_tool_error(self, error: Exception, **kwargs: Any) -> Any:
        self.events.append({"type": "tool_error", "error": str(error)})
        
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        self.events.append({"type": "llm_start", "prompts": prompts})
        
    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        self.events.append({"type": "llm_end"})

    def on_agent_action(self, action: Any, **kwargs: Any) -> Any:
        pass
    
    def on_agent_finish(self, finish: Any, **kwargs: Any) -> Any:
        pass

# =============================
# 🚀 App 시작 & 전역 설정
# =============================
load_dotenv()
st.set_page_config(page_title="DF Chatbot (Gemini)", page_icon="✨", layout="wide")
st.title("✨ DataFrame Chatbot (Gemini + LangChain)")
st.caption("두 CSV 비교 + 이상점 중심 EDA(원클릭) + SSD Telemetry 유틸")

# =============================
# 📁 데이터 로드 & 세션 상태
# =============================
DEFAULT_DATA_DIR = os.getenv("DATA_DIR", "/Users/najongseong/dataset")
DFB_DEFAULT_NAME = "telemetry_raw.csv"  # df_B 기본 파일명
SUPPORTED_EXTENSIONS = (".csv", ".parquet")

def _init_session_state():
    for key, default in [
        ("DATA_DIR", DEFAULT_DATA_DIR),
        ("df_A_data", None),
        ("df_A_name", "No Data"),
        ("csv_path", os.path.join(DEFAULT_DATA_DIR, "stormtrooper.csv")),
        ("df_B_data", None),
        ("df_B_name", "No Data"),
        ("csv_b_path", ""),
        ("explanation_lang", "English"),
        ("df_A_signature", ""),
        ("df_B_signature", ""),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

_init_session_state()

def _read_table(path: str) -> pd.DataFrame:
    """확장자에 맞게 CSV 또는 Parquet를 로드합니다."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"지원하지 않는 파일 형식입니다: {ext}")

def load_df_A(path: str, display_name: str):
    """지정된 경로에서 df_A를 로드하고 세션 상태를 업데이트."""
    try:
        new_df = _read_table(path)
        st.session_state["df_A_data"] = new_df
        st.session_state["df_A_name"] = display_name
        st.session_state["csv_path"] = path 
        return True, f"Loaded file: {display_name} (Shape: {new_df.shape})"
    except Exception as e:
        st.session_state["df_A_data"] = None
        st.session_state["df_A_name"] = "Load Failed"
        st.session_state["csv_path"] = "" 
        return False, f"데이터 로드 실패: {path}\n{e}"

def load_df_B(path: str, display_name: str):
    """지정된 경로에서 df_B를 로드하고 세션 상태를 업데이트."""
    try:
        new_df = _read_table(path)
        st.session_state["df_B_data"] = new_df
        st.session_state["df_B_name"] = display_name
        st.session_state["csv_b_path"] = path 
        return True, f"Loaded file (df_B): {display_name} (Shape: {new_df.shape})"
    except Exception as e:
        st.session_state["df_B_data"] = None
        st.session_state["df_B_name"] = "Load Failed"
        st.session_state["csv_b_path"] = "" 
        return False, f"df_B 로드 실패: {path}\n{e}"

# =============================
# 📚 사이드바: 폴더/파일 선택
# =============================
with st.sidebar:
    st.markdown("### 💬 EDA 설명 언어")
    lang_options = ["English", "한국어"]
    current_lang = st.session_state.get("explanation_lang", "English")
    selected_idx = lang_options.index(current_lang) if current_lang in lang_options else 0
    st.session_state["explanation_lang"] = st.selectbox(
        "Agent 요약 언어",
        options=lang_options,
        index=selected_idx,
    )

    st.markdown("### 🗂️ 1. 데이터 폴더 설정")
    new_data_dir = st.text_input("Enter Data Directory Path",
                                 value=st.session_state["DATA_DIR"],
                                 key="data_dir_input")
    if st.button("Set Directory"):
        if os.path.isdir(new_data_dir):
            st.session_state["DATA_DIR"] = new_data_dir
            st.session_state["df_A_data"] = None
            st.session_state["df_A_name"] = "No Data"
            st.session_state["csv_path"] = "" 
            st.session_state["df_B_data"] = None
            st.session_state["df_B_name"] = "No Data"
            st.session_state["csv_b_path"] = "" 
            st.success(f"Directory set to: `{new_data_dir}`")
            st.rerun()
        else:
            st.error(f"Invalid directory path: `{new_data_dir}`")

    DATA_DIR = st.session_state["DATA_DIR"]
    DFB_DEFAULT = os.path.join(DATA_DIR, DFB_DEFAULT_NAME)

    st.markdown("---")
    st.markdown("### 📄 2. df_A 파일 선택")
    st.caption(f"Search directory: `{DATA_DIR}`")
    data_files: List[str] = []
    try:
        if os.path.isdir(DATA_DIR):
            for f in os.listdir(DATA_DIR):
                if f.lower().endswith(SUPPORTED_EXTENSIONS):
                    data_files.append(f)
            data_files.sort()
        else:
            st.warning("유효한 데이터 디렉토리를 설정해주세요.")
    except Exception as e:
        st.error(f"폴더 접근 오류: {e}")

    selected_file = st.selectbox("Select data file for df_A",
                                 options=["--- Select a file ---"] + data_files,
                                 key="file_selector")
    if st.button("Load Selected File (df_A)"):
        if selected_file and selected_file != "--- Select a file ---":
            file_path = os.path.join(DATA_DIR, selected_file)
            success, message = load_df_A(file_path, selected_file)
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)
        else:
            st.warning("df_A 파일을 선택해주세요.")

    st.markdown("---")
    st.markdown("### 📄 3. df_B 파일 선택 (비교용)")
    selected_file_b = st.selectbox("Select data file for df_B",
                                   options=["--- Select a file ---"] + data_files,
                                   key="file_selector_b")
    if st.button("Load Selected File (df_B)"):
        if selected_file_b and selected_file_b != "--- Select a file ---":
            file_path_b = os.path.join(DATA_DIR, selected_file_b)
            success, message = load_df_B(file_path_b, selected_file_b)
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)
        else:
            st.warning("df_B 파일을 선택해주세요.")

    st.markdown("---")
    st.caption(f"**현재 로드 파일 경로(df_A):** `{st.session_state.get('csv_path', 'Not loaded')}`")
    st.caption(f"**현재 로드 파일 경로(df_B):** `{st.session_state.get('csv_b_path', 'Not loaded')}`")
    st.caption(f"df_B 기본 가정 파일: `{os.path.basename(DFB_DEFAULT)}`")

# 최종 df 결정
df_A: Optional[pd.DataFrame] = st.session_state["df_A_data"]
df_B: Optional[pd.DataFrame] = st.session_state["df_B_data"]

def _signature(df: Optional[pd.DataFrame], path: str) -> str:
    if df is None:
        return "none"
    rows, cols = df.shape
    return f"{path}|{rows}x{cols}"

current_sig_A = _signature(df_A, st.session_state.get("csv_path", ""))
prev_sig_A = st.session_state.get("df_A_signature", "")
dataset_changed = current_sig_A != prev_sig_A
st.session_state["df_A_signature"] = current_sig_A

current_sig_B = _signature(df_B, st.session_state.get("csv_b_path", ""))
prev_sig_B = st.session_state.get("df_B_signature", "")
df_b_changed = current_sig_B != prev_sig_B
st.session_state["df_B_signature"] = current_sig_B

if df_A is None:
    st.error("분석할 DataFrame (df_A)을 로드하지 못했습니다. 유효한 디렉토리와 지원되는 데이터 파일을 선택해주세요.")
    st.stop()

# =============================
# 🖼️ Preview
# =============================
st.subheader("Preview")
st.write(f"**Loaded file for df_A:** `{st.session_state['df_A_name']}` (Shape: {df_A.shape})")
st.dataframe(df_A.head(10), width="stretch")
if df_B is not None:
    with st.expander(f"df_B Preview — {st.session_state['df_B_name']} (Shape: {df_B.shape})", expanded=False):
        st.dataframe(df_B.head(10), width="stretch")

# =============================
# 💬 Chat history
# =============================
history = StreamlitChatMessageHistory(key="lc_msgs:dfchat")
if dataset_changed or df_b_changed:
    history.clear()

# =============================
# 🛠️ Tools 정의
# =============================
pytool = PythonAstREPLTool(
    globals={
        "pd": pd,
        "np": np,
        "plt": plt,
        "df": df_A,      # alias
        "df_A": df_A,    # main
        "df_B": df_B,    # secondary (optional)
        "df_join": None,
        "duckdb": duckdb,
    },
    name="python_repl_ast",
    description="Execute Python on df_A/df_B/df_join with pandas/matplotlib."
)

@tool
def load_loading_csv(filename: str) -> str:
    """Load a CSV or Parquet from DATA_DIR into 'loading_df'. Pass only file name."""
    current_data_dir = st.session_state.get("DATA_DIR", DEFAULT_DATA_DIR)
    path = os.path.join(current_data_dir, filename)
    try:
        new_df = _read_table(path)
    except Exception as e:
        return f"Failed to load {path}: {e}"
    pytool.globals["loading_df"] = new_df
    preview = new_df.head(10).to_markdown(index=False)
    shape = f"{new_df.shape[0]} rows x {new_df.shape[1]} cols"
    return f"Loaded {filename} from {current_data_dir} into loading_df\nShape: {shape}\n\nPreview (head):\n{preview}"

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
        current_df = pytool.globals.get("df_A")
        source = "df_A"
    if current_df is None:
        return "df_A not loaded."
    use_cols = [c.strip() for c in (cols or "").split(",") if c.strip()] or list(current_df.columns)
    missing = [c for c in use_cols if c not in current_df.columns]
    if missing:
        return f"Missing columns: {missing}\nAvailable: {list(current_df.columns)}"
    desc = current_df[use_cols].describe(include="all").transpose()
    shape = f"{current_df.shape[0]} rows x {current_df.shape[1]} cols"
    return f"[source={source} | shape={shape}]\n\n" + desc.to_markdown()

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
            f.savefig(img, format="png", dpi=110, bbox_inches="tight")
            img.seek(0)
            zf.writestr(f"plot_{i}.png", img.read())
    return f"Zipped {len(figs)} plots. Bytes={len(buf.getvalue())}"

@tool
def load_df_b(filename: str = "") -> str:
    """
    Load a CSV or Parquet into 'df_B'. If empty, defaults to telemetry_raw.csv under DATA_DIR.
    """
    current_data_dir = st.session_state.get("DATA_DIR", DEFAULT_DATA_DIR)
    if not filename:
        path = os.path.join(current_data_dir, DFB_DEFAULT_NAME)
        show_name = os.path.basename(path)
    else:
        path = os.path.join(current_data_dir, filename) if not os.path.isabs(filename) else filename
        show_name = os.path.basename(path)
    try:
        new_df = _read_table(path)
    except Exception as e:
        return f"Failed to load df_B from {path}: {e}"
    pytool.globals["df_B"] = new_df
    preview = new_df.head(10).to_markdown(index=False)
    return f"Loaded df_B from '{show_name}' (full: {path}) with shape {new_df.shape}\n\nPreview:\n{preview}"

@tool
def describe_columns_on(target: str = "A", cols: str = "") -> str:
    """
    Describe columns from df_A or df_B. target: 'A' | 'B'; cols: comma-separated; empty -> all
    """
    t = (target or "A").strip().upper()
    if t == "B":
        cur = pytool.globals.get("df_B")
        if cur is None:
            return "df_B is not loaded. Use load_df_b() or sidebar."
        source = "df_B"
    else:
        cur = pytool.globals.get("df_A")
        source = "df_A"
    if cur is None:
        return "df_A not loaded."
    use_cols = [c.strip() for c in (cols or "").split(",") if c.strip()] or list(cur.columns)
    missing = [c for c in use_cols if c not in cur.columns]
    if missing:
        return f"Missing columns: {missing}\nAvailable: {list(cur.columns)}"
    desc = cur[use_cols].describe(include="all").transpose()
    return f"[source={source} | shape={cur.shape}]\n\n" + desc.to_markdown()

@tool
def sql_on_dfs(query: str) -> str:
    """
    Run DuckDB SQL over df_A, df_B (if loaded), df_join (if created).
    Tables: df_A, df_B, df_join
    """
    try:
        if pytool.globals.get("df_A") is not None:
            duckdb.register("df_A", pytool.globals["df_A"])
        if pytool.globals.get("df_B") is not None:
            duckdb.register("df_B", pytool.globals["df_B"])
        if pytool.globals.get("df_join") is not None:
            duckdb.register("df_join", pytool.globals["df_join"])
        out = duckdb.sql(query).df()
        return out.head(200).to_markdown(index=False)
    except Exception as e:
        return f"SQL error: {e}"

@tool
def propose_join_keys() -> str:
    """Suggest same-name & compatible-dtype join key candidates between df_A and df_B."""
    A = pytool.globals.get("df_A")
    B = pytool.globals.get("df_B")
    if A is None:
        return "df_A not loaded."
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
        return "No obvious same-name & same-type keys. Consider casting or mapping."
    md = "| key | dtype |\n|---|---|\n" + "\n".join([f"| {k} | {t} |" for k, t in pairs])
    return f"Candidate join keys:\n{md}"

@tool
def align_time_buckets(target: str = "A", column: str = "datetime", freq: str = "H") -> str:
    """
    Resample time-like column to buckets and store as df_A_bucketed or df_B_bucketed.
    freq like 'H','D','15min'
    """
    t = (target or "A").strip().upper()
    cur = pytool.globals.get("df_B") if t == "B" else pytool.globals.get("df_A")
    if cur is None:
        return f"df_{t} not loaded."
    if column not in cur.columns:
        return f"Column '{column}' not in df_{t}."
    tmp = cur.copy()
    tmp[column] = pd.to_datetime(tmp[column], errors="coerce")
    bucket_col = f"{column}_bucket"
    tmp[bucket_col] = tmp[column].dt.to_period(freq).dt.to_timestamp()
    pytool.globals[f"df_{t}_bucketed"] = tmp
    prev = tmp[[bucket_col]].head().to_markdown(index=False)
    return f"Created df_{t}_bucketed with '{bucket_col}' at freq={freq}.\nPreview:\n{prev}"

# ---------- 비교 유틸 ----------
@tool
def compare_on_keys(keys: str, how: str = "inner", atol: Any = 0.0, rtol: Any = 0.0) -> str:
    """
    Join df_A & df_B on comma-separated `keys`, then compare shared columns.
    Robust to inputs like: "machineID,datetime" or "keys='machineID,datetime'".
    Creates global 'df_join'.
    """
    A = pytool.globals.get("df_A")
    B = pytool.globals.get("df_B")
    if A is None:
        return "df_A not loaded."
    if B is None:
        return "df_B is not loaded."

    # tol 파싱
    atol_val = _parse_float(atol, 0.0)
    rtol_val = _parse_float(rtol, 0.0)

    ks = (keys or "").strip()
    if ks.lower().startswith("keys="):
        ks = ks.split("=", 1)[1].strip()
    if (ks.startswith("'") and ks.endswith("'")) or (ks.startswith('"') and ks.endswith('"')):
        ks = ks[1:-1]
    key_cols = [k.strip() for k in ks.split(",") if k.strip()]
    if not key_cols:
        return "Please provide one or more keys (comma-separated)."
    for k in key_cols:
        if k not in A.columns or k not in B.columns:
            return f"Key '{k}' not found in both dataframes."

    shared_cols = [c for c in A.columns if c in B.columns and c not in key_cols]
    A_ = A[key_cols + shared_cols].copy()
    B_ = B[key_cols + shared_cols].copy()
    A_.columns = [*key_cols] + [f"{c}__A" for c in shared_cols]
    B_.columns = [*key_cols] + [f"{c}__B" for c in shared_cols]

    df_join = pd.merge(A_, B_, on=key_cols, how=how)
    pytool.globals["df_join"] = df_join
    try:
        duckdb.register("df_join", df_join)
    except Exception:
        pass

    numeric, categorical = [], []
    for c in shared_cols:
        a = df_join.get(f"{c}__A")
        b = df_join.get(f"{c}__B")
        if a is None or b is None:
            continue
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            diff = (a - b).astype("float64")
            eq = (diff.abs() <= (atol_val + rtol_val * b.abs().fillna(0))).fillna(False)
            numeric.append({
                "column": c,
                "count": int(diff.notna().sum()),
                "mean_A": float(a.mean(skipna=True)),
                "mean_B": float(b.mean(skipna=True)),
                "mean_diff": float(diff.mean(skipna=True)),
                "abs_mean_diff": float(diff.abs().mean(skipna=True)),
                "pct_equal_with_tol": float(eq.mean() * 100.0),
            })
        else:
            eq = (a.astype("string") == b.astype("string"))
            categorical.append({
                "column": c,
                "count": int(eq.notna().sum()),
                "match_rate_%": float(eq.mean(skipna=True) * 100.0),
                "n_unique_A": int(a.nunique(dropna=True)),
                "n_unique_B": int(b.nunique(dropna=True)),
            })

    out = [f"[compare_on_keys] how={how}, rows={len(df_join)}, keys={key_cols}, atol={atol_val}, rtol={rtol_val}"]
    if numeric:
        df_num = pd.DataFrame(numeric).sort_values("abs_mean_diff", ascending=False)
        out.append("**Numeric comparison (top 20 by abs_mean_diff):**\n" + df_num.head(20).to_markdown(index=False))
    else:
        out.append("**Numeric comparison:** None")
    if categorical:
        df_cat = pd.DataFrame(categorical).sort_values("match_rate_%")
        out.append("**Categorical comparison (lowest 20 match first):**\n" + df_cat.head(20).to_markdown(index=False))
    else:
        out.append("**Categorical comparison:** None")
    out.append("\nTip: 시각화가 필요하면 python_repl_ast에서 df_join을 사용하세요.")
    return "\n\n".join(out)

@tool
def mismatch_report(column: str, top_k: Any = 20) -> str:
    """On df_join, show largest numeric diffs or frequent categorical mismatches for a column."""
    dj = pytool.globals.get("df_join")
    if dj is None:
        return "df_join not found. Run compare_on_keys() first."

    topk_val = _parse_int(top_k, 20)

    colA = f"{column}__A"
    colB = f"{column}__B"
    if colA not in dj.columns or colB not in dj.columns:
        return f"Column '{column}' not found in df_join."
    a, b = dj[colA], dj[colB]
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        d = (a - b).abs()
        res = dj.assign(abs_diff=d).sort_values("abs_diff", ascending=False).head(topk_val)
        return res.to_markdown(index=False)
    neq = dj[a.astype("string") != b.astype("string")]
    if len(neq) == 0:
        return "No mismatches."
    counts = (
        neq[[colA, colB]]
        .astype("string")
        .value_counts()
        .reset_index(name="count")
        .head(topk_val)
    )
    return counts.to_markdown(index=False)

# ---------- SSD Telemetry EDA 유틸 ----------
@tool
def make_timesafe(column: str = "datetime", tz: str = "UTC") -> str:
    """
    Parse df_A/df_B[<column>] to datetime; if tz provided, localize/convert. Updates in-place.
    """
    changed = []
    for name in ["df_A", "df_B"]:
        cur = pytool.globals.get(name)
        if cur is None or column not in cur.columns:
            continue
        tmp = cur.copy()
        tmp[column] = pd.to_datetime(tmp[column], errors="coerce")
        if tz:
            try:
                if tmp[column].dt.tz is None:
                    tmp[column] = tmp[column].dt.tz_localize(tz)
                else:
                    tmp[column] = tmp[column].dt.tz_convert(tz)
            except Exception:
                pass
        pytool.globals[name] = tmp
        changed.append(f"{name}({len(tmp)} rows)")
    if not changed:
        return f"No target column '{column}' found in df_A/df_B."
    return f"[make_timesafe] column='{column}', tz='{tz}' → updated: {', '.join(changed)}"

@tool
def create_features(kind: str = "wear,temp,error,perf") -> str:
    """
    Create domain features on df_A if sources exist:
    - wear: rolling std etc.
    - temp: delta & rolling p95
    - error: per TB / per hour
    - perf: WAF, tail-latency ratio
    """
    A = pytool.globals.get("df_A")
    if A is None:
        return "df_A not loaded."
    out_cols = []
    dfw = A.copy()
    kinds = {k.strip() for k in (kind or "").split(",") if k.strip()}

    if "wear" in kinds and "wear_leveling_count" in dfw.columns:
        dfw["wear_leveling_std_5"] = dfw["wear_leveling_count"].rolling(5, min_periods=1).std()
        out_cols += ["wear_leveling_std_5"]

    if "temp" in kinds and "temperature" in dfw.columns:
        dfw["temp_delta"] = dfw["temperature"].diff()
        out_cols += ["temp_delta"]
        if "datetime" in dfw.columns:
            try:
                ts = pd.to_datetime(dfw["datetime"], errors="coerce")
                dfw = dfw.assign(__ts=ts).sort_values("__ts")
                dfw["temp_p95_12"] = (
                    dfw["temperature"].rolling(12, min_periods=3)
                    .apply(lambda x: np.nanpercentile(x, 95), raw=True)
                )
                dfw = dfw.drop(columns=["__ts"])
                out_cols += ["temp_p95_12"]
            except Exception:
                pass

    if "error" in kinds:
        if "uncorrectable_error_count" in dfw.columns and "tbw" in dfw.columns:
            safe_tbw = dfw["tbw"].replace(0, np.nan)
            dfw["uncorrectable_per_tb"] = dfw["uncorrectable_error_count"] / safe_tbw
            out_cols += ["uncorrectable_per_tb"]
        if "datetime" in dfw.columns and "uncorrectable_error_count" in dfw.columns:
            ts = pd.to_datetime(dfw["datetime"], errors="coerce")
            dfw = dfw.assign(__ts=ts).sort_values("__ts")
            delta = dfw["uncorrectable_error_count"].diff()
            dt_hours = dfw["__ts"].diff().dt.total_seconds() / 3600.0
            dfw["uncorr_per_hour"] = delta / dt_hours.replace(0, np.nan)
            dfw = dfw.drop(columns=["__ts"])
            out_cols += ["uncorr_per_hour"]

    if "perf" in kinds:
        if "nand_writes" in dfw.columns and "host_writes" in dfw.columns:
            denom = dfw["host_writes"].replace(0, np.nan)
            dfw["waf"] = dfw["nand_writes"] / denom
            out_cols += ["waf"]
        if all(c in dfw.columns for c in ["latency_p50", "latency_p99"]):
            denom = dfw["latency_p50"].replace(0, np.nan)
            dfw["latency_tail_ratio"] = dfw["latency_p99"] / denom
            out_cols += ["latency_tail_ratio"]

    if not out_cols:
        return "No feature created (missing source columns)."
    pytool.globals["df_A"] = dfw
    return f"[create_features] created: {out_cols}"

@tool
def rolling_stats(cols: str, window: str = "24H", on: str = "datetime") -> str:
    """
    Rolling mean/std for numeric cols in df_A using time window like '24H','7D'.
    Saves to pytool.globals['df_A_rolling'].
    """
    A = pytool.globals.get("df_A")
    if A is None:
        return "df_A not loaded."
    targets = [c.strip() for c in (cols or "").split(",") if c.strip()]
    if not targets:
        return "Please provide cols (comma-separated)."
    for c in targets:
        if c not in A.columns:
            return f"Column '{c}' not found in df_A."
    if on not in A.columns:
        return f"Time column '{on}' not found."
    tmp = A.copy()
    tmp[on] = pd.to_datetime(tmp[on], errors="coerce")
    tmp = tmp.sort_values(on).set_index(on)
    out = pd.DataFrame(index=tmp.index)
    for c in targets:
        if pd.api.types.is_numeric_dtype(tmp[c]):
            out[f"{c}_roll_mean"] = tmp[c].rolling(window, min_periods=3).mean()
            out[f"{c}_roll_std"] = tmp[c].rolling(window, min_periods=3).std()
    out = out.reset_index().rename(columns={on: on})
    pytool.globals["df_A_rolling"] = out
    return f"[rolling_stats] window={window}, cols={targets}"

@tool
def stl_decompose(col: str, period: Any = 24, on: str = "datetime") -> str:
    """
    STL decomposition on df_A[col] with seasonal period (e.g., 24 for hourly daily).
    Saves components to pytool.globals['df_A_stl'].
    """
    if STL is None:
        return "statsmodels가 설치되어 있지 않습니다. (pip install statsmodels)"
    A = pytool.globals.get("df_A")
    if A is None:
        return "df_A not loaded."

    # period 파싱
    period_val = _parse_int(period, 24)

    # 입력 유연화
    s = str(col).strip()
    try:
        if s.startswith("{") and s.endswith("}"):
            import json
            obj = json.loads(s)
            s = obj.get("col") or obj.get("column") or obj.get("name") or ""
    except Exception:
        pass
    for prefix in ("col=", "column=", "name="):
        if s.lower().startswith(prefix):
            s = s.split("=", 1)[1].strip()
            break
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        s = s[1:-1]

    if s not in A.columns:
        return f"Column '{s}' not found."
    if on not in A.columns:
        return f"Time column '{on}' not found."
    ts = pd.to_datetime(A[on], errors="coerce")
    x = pd.to_numeric(A[s], errors="coerce")
    ok = ts.notna() & x.notna()
    ts, x = ts[ok], x[ok]
    x = pd.Series(x.values, index=ts).sort_index()
    need = max(2 * period_val, 30)
    if len(x) < need:
        return f"Not enough points for STL (need ~{need})."
    res = STL(x, period=period_val, robust=True).fit()
    out = pd.DataFrame({
        on: x.index,
        f"{s}_trend": res.trend.values,
        f"{s}_seasonal": res.seasonal.values,
        f"{s}_resid": res.resid.values,
    })
    pytool.globals["df_A_stl"] = out
    return f"[stl_decompose] period={period_val}, col='{s}'"

@tool
def anomaly_iqr(col: str) -> str:
    """
    Mark IQR-based outliers on df_A[col]. Adds '{col}_is_outlier_iqr' boolean.
    Accepts inputs like:
      - host_total_write_gb
      - col=host_total_write_gb / column=host_total_write_gb / name=host_total_write_gb
      - {"col":"host_total_write_gb"} / {"column":"host_total_write_gb"}
    """
    A = pytool.globals.get("df_A")
    if A is None:
        return "df_A not loaded."

    s = str(col).strip()
    try:
        if s.startswith("{") and s.endswith("}"):
            import json
            obj = json.loads(s)
            s = obj.get("col") or obj.get("column") or obj.get("name") or ""
    except Exception:
        pass
    for prefix in ("col=", "column=", "name="):
        if s.lower().startswith(prefix):
            s = s.split("=", 1)[1].strip()
            break
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        s = s[1:-1]

    if s == "":
        return "Please provide a column name."
    if s not in A.columns:
        return f"Column '{s}' not found."

    x = pd.to_numeric(A[s], errors="coerce")
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    flags = (x < lo) | (x > hi)
    A[f"{s}_is_outlier_iqr"] = flags
    pytool.globals["df_A"] = A
    return f"[anomaly_iqr] col='{s}', bounds=({lo:.3f},{hi:.3f}), outliers={int(flags.sum())}"

@tool
def anomaly_isoforest(cols: str, contamination: Any = 0.01, random_state: Any = 42) -> str:
    """
    IsolationForest anomalies on df_A[cols]. Adds 'isoforest_outlier' boolean.
    Accepts:
      - "temperature,waf"
      - "cols=temperature,waf"
      - {"cols": "temperature,waf"}
    """
    if IsolationForest is None:
        return "scikit-learn이 설치되어 있지 않습니다. (pip install scikit-learn)"
    A = pytool.globals.get("df_A")
    if A is None:
        return "df_A not loaded."

    contamination_val = _parse_float(contamination, 0.01)
    rs_val = _parse_int(random_state, 42)

    raw_cols = cols
    targets: List[str] = []

    def _normalize_iterable(values) -> List[str]:
        return [str(c).strip() for c in values if str(c).strip()]

    if isinstance(raw_cols, (list, tuple, set)):
        targets = _normalize_iterable(raw_cols)
        s = ""
    else:
        s = str(raw_cols).strip()
        try:
            if s.startswith("{") and s.endswith("}"):
                import json
                obj = json.loads(s)
                extracted = obj.get("cols") or obj.get("columns") or obj.get("features") or ""
                if isinstance(extracted, (list, tuple, set)):
                    targets = _normalize_iterable(extracted)
                    s = ""
                else:
                    s = str(extracted or "").strip()
        except Exception:
            pass
        if not targets:
            for prefix in ("cols=", "columns=", "features="):
                if s.lower().startswith(prefix):
                    s = s.split("=", 1)[1].strip()
                    break
        if not targets:
            if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
                s = s[1:-1]
            targets = [c.strip() for c in s.split(",") if c.strip()]

    if not targets:
        return "Please provide cols."
    for c in targets:
        if c not in A.columns:
            return f"Column '{c}' not found."

    X = A[targets].apply(pd.to_numeric, errors="coerce").dropna()
    if X.shape[0] < 20:
        return "Not enough rows for IsolationForest (>=20 recommended)."
    clf = IsolationForest(n_estimators=200, contamination=contamination_val, random_state=rs_val)
    pred = clf.fit_predict(X.values)  # -1 = outlier
    flags = pd.Series(pred == -1, index=X.index)
    A["isoforest_outlier"] = False
    A.loc[flags.index, "isoforest_outlier"] = flags
    pytool.globals["df_A"] = A
    return f"[anomaly_isoforest] cols={targets}, contamination={contamination_val}, outliers={int(flags.sum())}"

@tool
def cohort_compare(by: str = "model,fw", agg: str = "mean", cols: str = "") -> str:
    """
    Group df_A by categorical columns and aggregate numeric metrics.
    Saves to pytool.globals['df_cohort'].
    """
    A = pytool.globals.get("df_A")
    if A is None:
        return "df_A not loaded."
    keys = [k.strip() for k in (by or "").split(",") if k.strip()]
    for k in keys:
        if k not in A.columns:
            return f"Group key '{k}' not found."
    if cols:
        metrics = [c.strip() for c in cols.split(",") if c.strip()]
    else:
        metrics = [c for c in A.columns if pd.api.types.is_numeric_dtype(A[c])]
    if not metrics:
        return "No numeric metrics to aggregate."
    aggfn = {"mean":"mean","median":"median","max":"max","min":"min","count":"count"}.get(agg.lower())
    if aggfn is None:
        return "Unsupported agg. Use mean|median|max|min|count."
    g = A.groupby(keys, dropna=False)[metrics].agg(aggfn).reset_index()
    pytool.globals["df_cohort"] = g
    return f"[cohort_compare] by={keys}, agg={agg}, metrics={metrics}\nPreview:\n{g.head().to_markdown(index=False)}"

@tool
def topn_machines(metric: str = "uncorrectable_per_tb", n: Any = 20, machine_col: str = "machineID") -> str:
    """
    List top-N machines in df_A by a metric (descending).
    """
    A = pytool.globals.get("df_A")
    if A is None:
        return "df_A not loaded."
    n_val = _parse_int(n, 20)
    if machine_col not in A.columns:
        return f"machine_col '{machine_col}' not found."
    if metric not in A.columns:
        return f"metric '{metric}' not found."
    sub = A[[machine_col, metric]].copy()
    sub = sub.sort_values(metric, ascending=False).head(n_val)
    pytool.globals["df_topN"] = sub
    return f"[topn_machines] metric='{metric}', n={n_val}\n{sub.to_markdown(index=False)}"

# ---------- 이상점 자동화 래퍼 / 보조 유틸 ----------
@tool
def select_numeric_candidates(min_unique: Any = 10, min_std: Any = 1e-9) -> str:
    """
    Return numeric candidate columns in df_A with >= min_unique and std > min_std.
    - min_unique: 빈 문자열/문자열 숫자/정수 모두 허용
    - min_std:    빈 문자열/문자열 숫자/실수 모두 허용
    """
    A = pytool.globals.get("df_A")
    if A is None:
        return "df_A not loaded."

    # ✅ 안전 파싱
    min_unique_val = _parse_int(min_unique, 10)
    min_std_val = _parse_float(min_std, 1e-9)

    nums = []
    for c in A.columns:
        if pd.api.types.is_numeric_dtype(A[c]):
            u = A[c].nunique(dropna=True)
            s = pd.to_numeric(A[c], errors="coerce").std(skipna=True)
            if u >= min_unique_val and (s is not None and s > min_std_val and np.isfinite(s)):
                nums.append(c)
    if not nums:
        return "No numeric candidates."
    return "Numeric candidates:\n" + pd.DataFrame({"column": nums}).to_markdown(index=False)

@tool
def rank_outlier_columns(method: str = "iqr_ratio", top_n: Any = 20) -> str:
    """
    Rank numeric columns by outlier ratio (IQR). Returns a table (head top_n).
    """
    A = pytool.globals.get("df_A")
    if A is None:
        return "df_A not loaded."

    top_n_val = _parse_int(top_n, 20)

    num_cols = [c for c in A.columns if pd.api.types.is_numeric_dtype(A[c]) and A[c].nunique(dropna=True) >= 10]
    rows = []
    for c in num_cols:
        x = pd.to_numeric(A[c], errors="coerce")
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr == 0:
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        rate = float(((x < lo) | (x > hi)).mean() * 100.0)
        rows.append({"column": c, "outlier_rate_%": rate, "lo": float(lo), "hi": float(hi)})
    if not rows:
        return "No IQR-detectable outliers."
    rank_df = pd.DataFrame(rows).sort_values("outlier_rate_%", ascending=False).head(top_n_val)
    pytool.globals["df_outlier_rank"] = rank_df
    return rank_df.to_markdown(index=False)

@tool
def plot_outliers(col: str, on: str = "datetime", sample: Any = 2000) -> str:
    """
    Plot boxplot and time series with IQR outlier highlighting for df_A[col].
    """
    A = pytool.globals.get("df_A")
    if A is None:
        return "df_A not loaded."
    sample_val = _parse_int(sample, 2000)
    s = str(col).strip()
    if s not in A.columns:
        return f"Column '{s}' not found."
    x = pd.to_numeric(A[s], errors="coerce")
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr == 0:
        return "IQR is zero or invalid."
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    flags = (x < lo) | (x > hi)

    # Boxplot
    plt.figure()
    pd.DataFrame({s: x}).plot(kind="box")
    plt.title(f"Boxplot: {s} (IQR bounds: {lo:.2f}, {hi:.2f})")
    plt.xlabel("")
    plt.tight_layout()

    # Timeseries (if available)
    if on in A.columns:
        ts = pd.to_datetime(A[on], errors="coerce")
        dfv = pd.DataFrame({on: ts, s: x, "__out": flags}).dropna().sort_values(on)
        if sample_val and len(dfv) > sample_val:
            step = max(1, len(dfv)//sample_val)
            dfv = dfv.iloc[::step, :]
        plt.figure()
        plt.plot(dfv[on], dfv[s], linestyle="-", marker="", alpha=0.7)
        ou = dfv[dfv["__out"]]
        if len(ou) > 0:
            plt.scatter(ou[on], ou[s], marker="o", s=12)
        plt.title(f"Timeseries: {s} (outliers highlighted)")
        plt.tight_layout()

    return f"[plot_outliers] col='{s}', outliers={int(flags.sum())}, bounds=({lo:.3f},{hi:.3f})"

@tool
def plot_outlier_overview(top_n: Any = 20) -> str:
    """
    컬럼별 IQR 이상치율을 막대 그래프로 요약 표시합니다.
    auto_outlier_eda 또는 rank_outlier_columns 결과가 없으면 즉석에서 계산합니다.
    """
    A = pytool.globals.get("df_A")
    if A is None:
        return "df_A not loaded."

    top_n_val = _parse_int(top_n, 20)

    rank_df = pytool.globals.get("df_outlier_rank")
    if rank_df is None or not isinstance(rank_df, pd.DataFrame) or "outlier_rate_%" not in rank_df.columns:
        rows = []
        for c in A.columns:
            if pd.api.types.is_numeric_dtype(A[c]) and A[c].nunique(dropna=True) >= 10:
                x = pd.to_numeric(A[c], errors="coerce")
                q1, q3 = x.quantile(0.25), x.quantile(0.75)
                iqr = q3 - q1
                if not np.isfinite(iqr) or iqr == 0:
                    continue
                lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                rate = float(((x < lo) | (x > hi)).mean() * 100.0)
                rows.append({"column": c, "outlier_rate_%": rate})
        if not rows:
            return "No IQR-detectable outliers."
        rank_df = pd.DataFrame(rows).sort_values("outlier_rate_%", ascending=False)
        pytool.globals["df_outlier_rank"] = rank_df

    top = rank_df.head(top_n_val)
    if top.empty:
        return "No IQR-detectable outliers."

    plt.figure(figsize=(8, max(3, 0.35 * len(top))))
    plt.barh(top["column"][::-1], top["outlier_rate_%"][::-1])
    plt.xlabel("Outlier rate (%)")
    plt.title(f"Top {min(top_n_val, len(top))} outlier columns (IQR)")
    plt.tight_layout()
    return f"[plot_outlier_overview] top_n={top_n_val}"

@tool
def plot_outliers_multi(cols: str, on: str = "datetime", sample: Any = 1500) -> str:
    """
    여러 수치 컬럼을 빠르게 훑어보는 소형 라인 플롯(스파크라인) 모음.
    IQR 경계 밖 점을 작은 마커로 표시합니다. 시간 컬럼이 있다면 그 축을 공유합니다.
    """
    A = pytool.globals.get("df_A")
    if A is None:
        return "df_A not loaded."

    targets = [c.strip() for c in (cols or "").split(",") if c.strip()]
    if not targets:
        return "Please provide cols (comma-separated)."
    for c in targets:
        if c not in A.columns:
            return f"Column '{c}' not found in df_A."

    ts = None
    if on in A.columns:
        ts = pd.to_datetime(A[on], errors="coerce")

    n = len(targets)
    sample_val = _parse_int(sample, 1500)
    fig, axes = plt.subplots(n, 1, figsize=(10, max(2.5, 1.5 * n)), sharex=ts is not None)
    if n == 1:
        axes = [axes]

    any_plotted = False
    for ax, c in zip(axes, targets):
        x = pd.to_numeric(A[c], errors="coerce")
        dfv = pd.DataFrame({c: x})
        if ts is not None:
            dfv[on] = ts
            dfv = dfv.dropna(subset=[on, c]).sort_values(on)
        else:
            dfv = dfv.dropna(subset=[c])

        if dfv.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_ylabel(c, rotation=0, labelpad=35, ha="right", va="center")
            continue

        if sample_val and len(dfv) > sample_val:
            step = max(1, len(dfv) // sample_val)
            dfv = dfv.iloc[::step, :]

        q1, q3 = dfv[c].quantile(0.25), dfv[c].quantile(0.75)
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr == 0:
            lo, hi = q1, q3
            flags = pd.Series(False, index=dfv.index)
        else:
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            flags = (dfv[c] < lo) | (dfv[c] > hi)

        if ts is not None:
            ax.plot(dfv[on], dfv[c], linewidth=1)
            out = dfv[flags]
            if len(out) > 0:
                ax.scatter(out[on], out[c], s=10)
        else:
            ax.plot(dfv.index, dfv[c], linewidth=1)
            out = dfv[flags]
            if len(out) > 0:
                ax.scatter(out.index, out[c], s=10)

        ax.set_ylabel(c, rotation=0, labelpad=35, ha="right", va="center")
        ax.grid(False)
        any_plotted = True

    if ts is not None:
        axes[-1].set_xlabel(on)
    if any_plotted:
        fig.suptitle("Outliers overview (IQR) — small multiples", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return f"[plot_outliers_multi] cols={targets}"

@tool
def plot_compare_timeseries(col: str, on: str = "datetime") -> str:
    """
    df_join에서 특정 컬럼의 A/B 시계열을 그리고, 절대 차이를 별도 축에 표시합니다.
    """
    dj = pytool.globals.get("df_join")
    if dj is None:
        return "df_join not found. Run compare_on_keys() first."

    colA, colB = f"{col}__A", f"{col}__B"
    if colA not in dj.columns or colB not in dj.columns:
        return f"Column '{col}' not found in df_join."
    if on not in dj.columns:
        return f"Time column '{on}' not found in df_join."

    dfv = dj[[on, colA, colB]].copy()
    dfv[on] = pd.to_datetime(dfv[on], errors="coerce")
    dfv[colA] = pd.to_numeric(dfv[colA], errors="coerce")
    dfv[colB] = pd.to_numeric(dfv[colB], errors="coerce")
    dfv = dfv.dropna().sort_values(on)
    if dfv.empty:
        return "No comparable rows with valid timestamps."

    dfv["abs_diff"] = (dfv[colA] - dfv[colB]).abs()

    plt.figure(figsize=(10, 4))
    plt.plot(dfv[on], dfv[colA], label="A")
    plt.plot(dfv[on], dfv[colB], label="B", alpha=0.8)
    plt.title(f"{col}: A vs B")
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(10, 2.8))
    plt.plot(dfv[on], dfv["abs_diff"])
    plt.title(f"{col}: |A - B|")
    plt.tight_layout()
    return f"[plot_compare_timeseries] col='{col}'"

@tool
def stl_plot(col: str, on: str = "datetime") -> str:
    """
    stl_decompose 이후 df_A_stl에 저장된 trend/seasonal/resid 시계열을 한 번에 시각화합니다.
    """
    stl_df = pytool.globals.get("df_A_stl")
    if stl_df is None or not isinstance(stl_df, pd.DataFrame) or stl_df.empty:
        return "Run stl_decompose first."
    if on not in stl_df.columns:
        return f"Time column '{on}' not found in df_A_stl."
    components = [f"{col}_trend", f"{col}_seasonal", f"{col}_resid"]
    missing = [c for c in components if c not in stl_df.columns]
    if missing:
        return f"Missing STL components: {missing}"
    chart = stl_df.copy()
    chart[on] = pd.to_datetime(chart[on], errors="coerce")
    chart = chart.dropna(subset=[on]).sort_values(on)
    if chart.empty:
        return "No valid timestamps in df_A_stl."
    plt.figure(figsize=(10, 5))
    for comp, label in zip(components, ["trend", "seasonal", "resid"]):
        plt.plot(chart[on], chart[comp], label=label)
    plt.title(f"STL components: {col}")
    plt.legend()
    plt.tight_layout()
    return f"[stl_plot] col='{col}'"

@tool
def corr_heatmap(cols: str = "") -> str:
    """
    선택한 컬럼(또는 전체 수치형)의 상관계수 히트맵을 렌더링합니다.
    """
    A = pytool.globals.get("df_A")
    if A is None:
        return "df_A not loaded."

    targets = [c.strip() for c in (cols or "").split(",") if c.strip()]
    if targets:
        for c in targets:
            if c not in A.columns:
                return f"Column '{c}' not found."
        dfv = A[targets].apply(pd.to_numeric, errors="coerce")
    else:
        dfv = A.select_dtypes(include=[np.number])

    if dfv.shape[1] < 2:
        return "Need at least 2 numeric columns."

    corr = dfv.corr(numeric_only=True)
    plt.figure(figsize=(max(5, 0.6 * corr.shape[1]), max(4, 0.6 * corr.shape[0])))
    plt.imshow(corr, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    plt.xticks(range(corr.shape[1]), corr.columns, rotation=90)
    plt.yticks(range(corr.shape[0]), corr.index)
    plt.colorbar(label="corr")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    return "[corr_heatmap] done"

@tool
def auto_outlier_eda(top_n: Any = 10, on: str = "datetime") -> str:
    """
    Run an outlier-first EDA pipeline on df_A:
    - select numeric candidates
    - anomaly_iqr on each → rank by outlier ratio
    - optional stl_decompose on top-1 if time column exists
    - optional anomaly_isoforest on k-best (if sklearn installed)
    Returns a concise ranked report (head 20).
    """
    A = pytool.globals.get("df_A")
    if A is None:
        return "df_A not loaded."

    top_n_val = _parse_int(top_n, 10)

    # 1) 후보 컬럼
    num_cols = [c for c in A.columns if pd.api.types.is_numeric_dtype(A[c])]
    num_cols = [c for c in num_cols if A[c].nunique(dropna=True) >= 10]
    if not num_cols:
        return "No numeric candidates."

    # 2) IQR 스캔
    rows = []
    for c in num_cols:
        x = pd.to_numeric(A[c], errors="coerce")
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr == 0:
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        flags = (x < lo) | (x > hi)
        rate = float(flags.mean() * 100.0)
        rows.append({"column": c, "outlier_rate_%": rate, "lo": float(lo), "hi": float(hi)})
    if not rows:
        return "No IQR-detectable outliers."
    rank_df = pd.DataFrame(rows).sort_values("outlier_rate_%", ascending=False)
    top_cols = rank_df.head(max(1, min(top_n_val, len(rank_df))))["column"].tolist()
    pytool.globals["df_outlier_rank"] = rank_df

    summary = ["[auto_outlier_eda] IQR scan complete.",
               "**Top outlier columns:**\n" + rank_df.head(20).to_markdown(index=False)]

    # 3) STL (선택)
    if on in A.columns and len(top_cols) > 0 and STL is not None:
        try:
            col = top_cols[0]
            ts = pd.to_datetime(A[on], errors="coerce")
            y = pd.to_numeric(A[col], errors="coerce")
            ok = ts.notna() & y.notna()
            ts, y = ts[ok], y[ok]
            y = pd.Series(y.values, index=ts).sort_index()
            if len(y) >= 48:
                res = STL(y, period=24, robust=True).fit()
                resid = pd.Series(res.resid, index=y.index)
                max_abs = float(resid.abs().max())
                when = str(resid.abs().idxmax())
                summary.append(f"**STL residual spike** for '{col}': max |resid|={max_abs:.3f} at {when}")
        except Exception:
            pass

    # 4) 다변량 (선택)
    if IsolationForest is not None:
        try:
            k = min(4, len(top_cols))
            if k >= 2:
                X = A[top_cols[:k]].apply(pd.to_numeric, errors="coerce").dropna()
                if len(X) >= 50:
                    clf = IsolationForest(n_estimators=200, contamination=0.02, random_state=42).fit(X.values)
                    out_rate = float((clf.predict(X.values) == -1).mean() * 100.0)
                    summary.append(f"**IsolationForest** on {top_cols[:k]} → outlier_rate≈{out_rate:.1f}%")
        except Exception:
            pass

    return "\n\n".join(summary)

# 툴 목록
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
    compare_on_keys,
    mismatch_report,
    make_timesafe,
    create_features,
    rolling_stats,
    stl_decompose,
    anomaly_iqr,
    anomaly_isoforest,
    cohort_compare,
    topn_machines,
    select_numeric_candidates,
    rank_outlier_columns,
    plot_outliers,
    plot_outlier_overview,
    plot_outliers_multi,
    plot_compare_timeseries,
    stl_plot,
    corr_heatmap,
    auto_outlier_eda,
]

# =============================
# 🤖 LLM 설정
# =============================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY가 설정되어 있지 않습니다. .env 또는 환경변수를 확인하세요.")
    st.stop()

llm = ChatGoogleGenerativeAI(
    # model="gemini-2.5-flash",
    model="gemini-2.5-flash-lite",
    api_key=GOOGLE_API_KEY,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    disable_streaming=False,  # streaming 경고 회피
)

# =============================
# 📜 Prompt (시스템 프롬프트 업데이트)
# =============================
headA = df_A.head().to_string(index=False)
headB = df_B.head().to_string(index=False) if df_B is not None else "(df_B not loaded)"

react_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert data analyst for SSD telemetry and tabular data. "
     "You work with two dataframes: df_A (main, alias: df) and df_B (optional, for comparison/labels).\n\n"
     "When the user asks for outlier-focused EDA, DO NOT ask questions. Immediately run this pipeline:\n"
     "  1) make_timesafe(column='datetime', tz='UTC') if 'datetime' exists\n"
     "  2) describe_columns → select_numeric_candidates → rank_outlier_columns\n"
     "  3) anomaly_iqr on top-N columns\n"
     "  4) stl_decompose on top-1 if time series available\n"
     "  5) anomaly_isoforest on k-best numeric (if sklearn available)\n"
     "  6) cohort_compare(by='model,fw', agg='mean') if columns exist\n"
     "  7) Summarize: top outlier columns + time spikes + cohort outliers + next steps.\n"
     "If df_B is loaded and user mentions comparison, use propose_join_keys then compare_on_keys.\n\n"
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
     "Tool routing guide:\n"
     "- Schema/summary → describe_columns, describe_columns_on\n"
     "- File load → load_loading_csv, load_df_b\n"
     "- SQL/join/aggregation → sql_on_dfs\n"
     "- TWO-CSV comparison → propose_join_keys → compare_on_keys('machineID,datetime') → mismatch_report('...')\n"
     "- SSD utilities → make_timesafe, create_features, rolling_stats, stl_decompose, anomaly_iqr, anomaly_isoforest, cohort_compare, topn_machines\n"
     "- Outlier one-click → auto_outlier_eda, then plot_outliers on top columns\n"
     "- Custom compute/plots → python_repl_ast (do complex tasks in ONE call and print results)\n\n"
     "For tools that take a column name (e.g., anomaly_iqr), pass ONLY the raw column name (e.g., temperature), "
     "not 'column=temperature'. If you accidentally wrote 'column=...', immediately continue by outputting just the raw name.\n\n"
     "Call compare_on_keys with just the keys string (e.g., 'machineID,datetime') or as JSON {{\"keys\":\"machineID,datetime\"}}.\n"
     "Do NOT pass \"keys='...'\" literal unless JSON.\n\n"
     f"df_A.head():\n{headA}\n\n"
     f"df_B.head():\n{headB}\n"
    ),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    ("assistant", "{agent_scratchpad}"),
])

# {tools}/{tool_names} 주입
tool_desc = render_text_description(tools)
tool_names = ", ".join([t.name for t in tools])
react_prompt = react_prompt.partial(tools=tool_desc, tool_names=tool_names)

# =============================
# ⚙️ ReAct Agent
# =============================
react_runnable = create_react_agent(llm, tools, prompt=react_prompt)
agent = AgentExecutor(
    agent=react_runnable,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True,
    max_iterations=20,
    early_stopping_method="generate",
    handle_parsing_errors=(
        "PARSING ERROR. DO NOT APOLOGIZE. Immediately continue by outputting ONLY:\n"
        "Action: describe_columns\n"
        "Action Input: \n"
    ),
)

# =============================
# 🔄 RunnableWithMessageHistory
# =============================
history = StreamlitChatMessageHistory(key="lc_msgs:dfchat")
agent_with_history = RunnableWithMessageHistory(
    agent,
    lambda session_id: history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# =============================
# 💻 UI 실행
# =============================
st.write("---")
user_q = st.chat_input(
    "예) 이상점 EDA 해줘 / auto_outlier_eda() / plot_outliers('temperature') / "
    "propose_join_keys / compare_on_keys('machineID,datetime') / mismatch_report('temperature') / "
    "rolling_stats(cols='temperature,uncorrectable_error_count', window='24H') / stl_decompose('temperature', 24)"
)

if user_q:
    left, right = st.columns([1, 1])
    with left:
        st.subheader("실시간 실행 로그")
        st_cb = StreamlitCallbackHandler(st.container())
    collector = SimpleCollectCallback()

    with st.spinner("Thinking with Gemini..."):
        try:
            result = agent_with_history.invoke(
                {"input": user_q},
                {
                    "callbacks": [st_cb, collector, StdOutCallbackHandler()],
                    "configurable": {"session_id": "two_csv_compare_and_ssd_eda"},
                }
            )
        except Exception as e:
            st.error(f"Agent 실행 중 오류: {e}")
            result = {"output": f"Agent 실행 중 오류: {e}"}

    st.success("Done.")
    final = result.get("output", "Agent가 최종 답변을 생성하지 못했습니다.")
    with right:
        st.subheader("Answer")
        final_text = final if isinstance(final, str) else str(final)
        lang_choice = st.session_state.get("explanation_lang", "English")
        final_display = final_text
        if lang_choice == "한국어" and final_text.strip():
            try:
                translation_prompt = (
                    "다음 분석 결과를 자연스럽고 간결한 한국어로 설명해줘.\n\n"
                    f"{final_text}"
                )
                translated_msg = llm.invoke(translation_prompt)
                translated_text = getattr(translated_msg, "content", None)
                if translated_text:
                    final_display = translated_text
            except Exception as e:
                st.warning(f"한국어 번역 중 오류가 발생했습니다: {e}")
        st.write(final_display)

        st.markdown("---")
        st.subheader("EDA Visualizations")
        visuals_rendered = False

        outlier_rank_df = pytool.globals.get("df_outlier_rank")
        if isinstance(outlier_rank_df, pd.DataFrame) and not outlier_rank_df.empty:
            if {"column", "outlier_rate_%"} <= set(outlier_rank_df.columns):
                visuals_rendered = True
                top_outliers = outlier_rank_df.head(15).set_index("column")["outlier_rate_%"]
                st.markdown("**Top Outlier Columns (IQR %)**")
                st.bar_chart(top_outliers)

        stl_df = pytool.globals.get("df_A_stl")
        if isinstance(stl_df, pd.DataFrame) and not stl_df.empty:
            stl_chart = stl_df.copy()
            time_col = stl_chart.columns[0]
            if time_col in stl_chart.columns:
                stl_chart[time_col] = pd.to_datetime(stl_chart[time_col], errors="coerce")
                stl_chart = stl_chart.dropna(subset=[time_col]).set_index(time_col).sort_index()
                numeric_cols = [c for c in stl_chart.columns if pd.api.types.is_numeric_dtype(stl_chart[c])]
                if numeric_cols:
                    visuals_rendered = True
                    st.markdown("**STL Decomposition Components**")
                    st.line_chart(stl_chart[numeric_cols])

        rolling_df = pytool.globals.get("df_A_rolling")
        if isinstance(rolling_df, pd.DataFrame) and not rolling_df.empty:
            roll_chart = rolling_df.copy()
            time_col = roll_chart.columns[0]
            if time_col in roll_chart.columns:
                roll_chart[time_col] = pd.to_datetime(roll_chart[time_col], errors="coerce")
                roll_chart = roll_chart.dropna(subset=[time_col]).set_index(time_col).sort_index()
                metric_cols = [c for c in roll_chart.columns if pd.api.types.is_numeric_dtype(roll_chart[c])]
                if metric_cols:
                    visuals_rendered = True
                    st.markdown("**Rolling Statistics**")
                    st.line_chart(roll_chart[metric_cols])

        topn_df = pytool.globals.get("df_topN")
        if isinstance(topn_df, pd.DataFrame) and not topn_df.empty:
            visuals_rendered = True
            st.markdown("**Top-N Entities**")
            st.dataframe(topn_df, use_container_width=True)

        current_df_a = pytool.globals.get("df_A")
        if isinstance(current_df_a, pd.DataFrame) and not current_df_a.empty:
            outlier_cols = [c for c in current_df_a.columns if c.endswith("_is_outlier_iqr")]
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
                            .apply(lambda s: int(s.fillna(False).sum()))
                            .rename("outlier_count")
                            .to_frame()
                        )
                        st.table(outlier_summary)

        if not visuals_rendered:
            st.caption("아직 표시할 EDA 시각화가 없습니다. auto_outlier_eda(), stl_decompose(), rolling_stats() 등을 먼저 실행해보세요.")

        # 현재까지 생성된 matplotlib 플롯 렌더링
        figs = [plt.figure(n) for n in plt.get_fignums()]
        for f in figs:
            f.set_size_inches(6, 4)
            f.set_dpi(100)
            st.pyplot(f, width="stretch")
        plt.close("all")

    # intermediate steps (디버깅 정보)xf
    st.write("---")
    st.subheader("intermediate_steps (툴 실행 상세)")
    steps = result.get("intermediate_steps", [])
    if steps:
        for i, (action, observation) in enumerate(steps, 1):
            with st.expander(f"Step {i}: {action.tool}"):
                st.markdown("**tool_input**")
                st.code(str(action.tool_input))
                st.markdown("**observation**")
                obs_txt = observation if isinstance(observation, str) else str(observation)
                st.code(obs_txt[:4000])
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
        with st.sidebar:
            st.chat_message("assistant").write(msg.content[:50] + "...")
