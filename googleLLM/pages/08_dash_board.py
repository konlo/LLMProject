import os
import io
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="CSV Dashboard", layout="wide")

st.title("📊 CSV Dashboard")
st.caption("Interactive, matplotlib-based dashboard for exploring a CSV file.")

# ---- File loading ----
default_path = "/Users/najongseong/dataset/playground-series-s5e8/train.csv"
st.sidebar.header("⚙️ Settings")
file_source = st.sidebar.radio("Data source", ["Use default path", "Upload file"])

df = None
load_error = None

def safe_read_csv(path, **kwargs):
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as e:
        return e

if file_source == "Use default path":
    if default_path and os.path.exists(default_path):
        preview = pd.read_csv(default_path, nrows=200)
        date_cols_guess = [c for c in preview.columns if any(k in c.lower() for k in ["date","time","timestamp"])]
        parse_dates = date_cols_guess if date_cols_guess else None
        result = safe_read_csv(default_path, parse_dates=parse_dates, low_memory=False)
        if isinstance(result, Exception):
            load_error = result
        else:
            df = result
    else:
        load_error = FileNotFoundError("Default CSV not found. Upload a file.")
else:
    up = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if up is not None:
        temp_preview = pd.read_csv(up, nrows=200)
        date_cols_guess = [c for c in temp_preview.columns if any(k in c.lower() for k in ["date","time","timestamp"])]
        up.seek(0)
        try:
            df = pd.read_csv(up, parse_dates=date_cols_guess if date_cols_guess else None, low_memory=False)
        except Exception as e:
            load_error = e

if load_error is not None:
    st.error(f"Failed to load CSV: {load_error}")
    st.stop()

if df is None or df.empty:
    st.warning("No data loaded yet.")
    st.stop()

# ---- Column typing ----
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = [c for c in df.columns if c not in numeric_cols]

# ---- Overview ----
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.metric("Rows", f"{len(df):,}")
with col_b:
    st.metric("Columns", f"{df.shape[1]:,}")
with col_c:
    st.metric("Missing values", f"{int(df.isna().sum().sum()):,}")

with st.expander("🔎 Preview data", expanded=True):
    st.dataframe(df.head(200), use_container_width=True)

# ---- Sidebar filters ----
st.sidebar.subheader("🔍 Quick filters")
filterable_cols = st.sidebar.multiselect("Choose columns to filter", categorical_cols[:20])
filtered_df = df.copy()
for col in filterable_cols:
    vals = ["(Missing)"] + sorted([str(v) for v in filtered_df[col].dropna().unique()])[:200]
    pick = st.sidebar.multiselect(col, vals)
    if pick:
        mask = filtered_df[col].astype(str).isin([p for p in pick if p != "(Missing)"])
        if "(Missing)" in pick:
            mask |= filtered_df[col].isna()
        filtered_df = filtered_df[mask]

st.markdown("### 📈 Distributions")
num_col = st.selectbox("Numeric column for histogram", numeric_cols, index=0 if numeric_cols else None)
if num_col:
    bins = st.slider("Bins", min_value=5, max_value=100, value=30, step=5)
    fig = plt.figure()
    plt.hist(filtered_df[num_col].dropna(), bins=bins)
    plt.xlabel(num_col)
    plt.ylabel("Count")
    plt.title(f"Histogram of {num_col}")
    st.pyplot(fig)

st.markdown("### 📦 Box plot by category")
cat_for_box = st.selectbox("Category", categorical_cols, index=0 if categorical_cols else None)
num_for_box = st.selectbox("Numeric", numeric_cols, index=0 if numeric_cols else None, key="box_num")
if cat_for_box and num_for_box:
    top_cats = filtered_df[cat_for_box].astype(str).value_counts().head(10).index.tolist()
    sub = filtered_df[filtered_df[cat_for_box].astype(str).isin(top_cats)][[cat_for_box, num_for_box]].dropna()
    fig2 = plt.figure()
    groups = [grp[num_for_box].values for _, grp in sub.groupby(cat_for_box)]
    plt.boxplot(groups, labels=[str(c) for c in sub[cat_for_box].astype(str).unique()])
    plt.xlabel(cat_for_box)
    plt.ylabel(num_for_box)
    plt.title(f"Box plot of {num_for_box} by {cat_for_box} (Top 10 categories)")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig2)

st.markdown("### 🔁 Scatter")
x_col = st.selectbox("X", numeric_cols, index=0 if numeric_cols else None, key="scatter_x")
y_col = st.selectbox("Y", numeric_cols, index=1 if len(numeric_cols) > 1 else 0 if numeric_cols else None, key="scatter_y")
if x_col and y_col and x_col != y_col:
    fig3 = plt.figure()
    plt.scatter(filtered_df[x_col].values, filtered_df[y_col].values, s=8, alpha=0.7)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"Scatter: {x_col} vs {y_col}")
    st.pyplot(fig3)

if numeric_cols:
    st.markdown("### 🔗 Correlation (numeric)")
    corr = filtered_df[numeric_cols].corr(numeric_only=True)
    fig4 = plt.figure()
    plt.imshow(corr, aspect="auto")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.colorbar()
    plt.title("Correlation heatmap")
    st.pyplot(fig4)

time_candidates = [c for c in df.columns if str(df[c].dtype).startswith("datetime64")]
if time_candidates:
    st.markdown("### 🕒 Time series")
    tcol = st.selectbox("Datetime column", time_candidates, index=0)
    y_ts = st.selectbox("Y (numeric)", numeric_cols, index=0 if numeric_cols else None, key="ts_y")
    if tcol and y_ts:
        ts = filtered_df[[tcol, y_ts]].dropna()
        ts = ts.sort_values(tcol)
        if len(ts) > 100_000:
            ts = ts.iloc[:: max(1, len(ts)//100_000)]
        fig5 = plt.figure()
        plt.plot(ts[tcol].values, ts[y_ts].values)
        plt.xlabel(tcol)
        plt.ylabel(y_ts)
        plt.title(f"Time series of {y_ts} over {tcol}")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig5)

st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit + matplotlib")
