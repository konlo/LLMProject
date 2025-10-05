"""Streamlit app for telemetry data chatbot with anomaly detection."""

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

SAMPLE_DATASETS: Dict[str, Path] = {
    "Telemetry (raw)": Path("dataset/telemetry_raw.csv"),
    "Telemetry (report)": Path("dataset/telemetry_report.csv"),
    "Ride bookings": Path("dataset/ncr_ride_bookings.csv"),
}

BASE_SYSTEM_PROMPT = """
You are a telemetry data analyst helping users explore and understand a dataset.
The user interacts with you via a Streamlit chatbot. Follow these rules:
1. Use the dataset context to ground your answers.
2. Reference anomaly detection findings when helpful and explain what the anomaly score means.
3. Suggest appropriate visual analyses that the user can perform in the dashboard when relevant.
4. If information is unavailable, clearly say so instead of guessing.

Here is the dataset context:
{context}
""".strip()


@st.cache_data(show_spinner="Loading dataset...")
def load_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner="Parsing uploaded file...")
def load_uploaded_dataset(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(file_bytes))


def coerce_datetime_columns(dataset: pd.DataFrame) -> pd.DataFrame:
    converted = dataset.copy()
    for column in converted.columns:
        if converted[column].dtype == "object" and (
            "time" in column.lower() or "date" in column.lower()
        ):
            try:
                converted[column] = pd.to_datetime(converted[column])
            except (ValueError, TypeError):
                continue
    return converted


@st.cache_data
def run_anomaly_detection(
    data: pd.DataFrame,
    features: List[str],
    contamination: float,
) -> pd.DataFrame:
    numeric_data = data[features].dropna()
    if numeric_data.empty:
        return pd.DataFrame(columns=["anomaly", "score"])  # type: ignore[return-value]

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(numeric_data.values)

    detector = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=200,
    )
    detector.fit(scaled_values)
    predictions = detector.predict(scaled_values)
    scores = detector.decision_function(scaled_values)

    result = pd.DataFrame(index=numeric_data.index)
    result["anomaly"] = predictions == -1
    result["score"] = -scores
    return result


def build_dataset_summary(
    dataset: Optional[pd.DataFrame],
    anomalies: Optional[pd.DataFrame],
    selected_features: List[str],
    contamination: float,
) -> str:
    if dataset is None or dataset.empty:
        return "No dataset is currently loaded."

    summary_lines = [
        f"Rows: {len(dataset):,}",
        f"Columns: {len(dataset.columns)}",
        "Columns and dtypes:",
    ]

    for col, dtype in dataset.dtypes.items():
        summary_lines.append(f"  - {col}: {dtype}")

    numeric_subset = dataset.select_dtypes(include=["number"])  # type: ignore[arg-type]
    if numeric_subset.shape[1] == 0:
        summary_lines.append("\nNumeric summary statistics: no numeric columns detected.")
    else:
        numeric_summary = numeric_subset.describe().transpose()
        summary_lines.append("\nNumeric summary statistics (mean ± std):")
        for col, stats in numeric_summary.iterrows():
            mean_value = stats.get("mean", np.nan)
            std_value = stats.get("std", np.nan)
            summary_lines.append(
                f"  - {col}: mean={mean_value:.3f}, std={std_value:.3f}"
            )

    if anomalies is not None and not anomalies.empty:
        total_anomalies = anomalies["anomaly"].sum()
        summary_lines.append(
            f"\nIsolationForest detected {total_anomalies} anomalies "
            f"({total_anomalies / len(anomalies) * 100:.2f}% of evaluated rows)"
        )
        top_anomalies = (
            anomalies[anomalies["anomaly"]]
            .sort_values("score", ascending=False)
            .head(5)
        )
        if not top_anomalies.empty:
            summary_lines.append("Top anomalous indices and scores:")
            for idx, row in top_anomalies.iterrows():
                summary_lines.append(f"  - index {idx}: score={row['score']:.4f}")
    else:
        summary_lines.append(
            "\nIsolationForest did not identify anomalies with the current configuration."
        )

    if selected_features:
        summary_lines.append(
            "\nAnomaly detection configuration: "
            f"features={', '.join(selected_features)}; "
            f"contamination={contamination:.3f}"
        )

    return "\n".join(summary_lines)


def render_anomaly_visualisation(
    dataset: pd.DataFrame,
    anomalies: Optional[pd.DataFrame],
    time_column: Optional[str],
    metric_column: str,
) -> None:
    plot_data = dataset.copy()
    if anomalies is not None and not anomalies.empty:
        plot_data = plot_data.join(anomalies[["anomaly"]], how="left")
        plot_data["anomaly"] = plot_data["anomaly"].fillna(False)
    else:
        plot_data["anomaly"] = False

    plot_data["anomaly_label"] = np.where(plot_data["anomaly"], "Anomaly", "Normal")

    if time_column:
        x_axis = time_column
        plot_data = plot_data.sort_values(time_column)
    else:
        plot_data = plot_data.reset_index().rename(columns={"index": "row_index"})
        x_axis = "row_index"

    fig = px.scatter(
        plot_data,
        x=x_axis,
        y=metric_column,
        color="anomaly_label",
        color_discrete_map={"Anomaly": "#d62728", "Normal": "#1f77b4"},
        title=f"{metric_column} with anomaly highlights",
        hover_data=plot_data.columns,
    )
    fig.update_traces(marker={"size": 9})
    st.plotly_chart(fig, use_container_width=True)


def prepare_chat_model() -> Optional[ChatGoogleGenerativeAI]:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    try:
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=api_key,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            streaming=True,
        )
    except Exception:
        return None


def main() -> None:
    st.set_page_config(
        page_title="Telemetry Analyst Chatbot",
        page_icon="🤖",
        layout="wide",
    )
    st.title("Telemetry Data Analyst Chatbot")
    st.markdown(
        """
        이 앱은 텔레메트리(telemetry) 데이터 분석을 도와주는 챗봇입니다.
        좌측 사이드바에서 데이터를 선택하거나 업로드하고, 이상치 탐지를 실행한 뒤 챗봇에게 질문해보세요.
        """
    )

    st.sidebar.header("데이터 선택")
    dataset_choice = st.sidebar.selectbox(
        "샘플 데이터",
        options=["사용자 업로드"] + list(SAMPLE_DATASETS.keys()),
    )

    uploaded_file = None
    if dataset_choice == "사용자 업로드":
        uploaded_file = st.sidebar.file_uploader(
            "CSV 파일 업로드",
            type="csv",
            help="업로드 후 자동으로 로딩됩니다.",
        )

    dataset: Optional[pd.DataFrame] = None
    if uploaded_file is not None:
        dataset = load_uploaded_dataset(uploaded_file.getvalue())
    elif dataset_choice != "사용자 업로드":
        dataset_path = SAMPLE_DATASETS[dataset_choice]
        if dataset_path.exists():
            dataset = load_dataset(dataset_path)
        else:
            st.sidebar.error(f"샘플 데이터 {dataset_path} 를 찾을 수 없습니다.")

    if dataset is None or dataset.empty:
        st.info("데이터가 로딩되면 미리보기와 분석 결과가 표시됩니다.")
        st.stop()

    dataset = coerce_datetime_columns(dataset)
    st.caption(f"현재 분석 중인 데이터셋: {dataset_choice}")

    st.subheader("데이터 미리보기")
    st.dataframe(dataset.head(100))

    numeric_columns = dataset.select_dtypes(include=["number"]).columns.tolist()
    st.sidebar.header("이상 탐지 설정")
    if not numeric_columns:
        st.sidebar.warning("수치형 컬럼이 없어 이상 탐지를 실행할 수 없습니다.")
        selected_features: List[str] = []
        anomalies = None
    else:
        default_selection = numeric_columns[: min(len(numeric_columns), 3)]
        selected_features = st.sidebar.multiselect(
            "이상 탐지에 사용할 수치형 컬럼",
            options=numeric_columns,
            default=default_selection,
        )
        contamination = st.sidebar.slider(
            "Contamination (이상치 비율 예상)",
            min_value=0.01,
            max_value=0.2,
            value=0.05,
            step=0.01,
        )
        anomalies = None
        if selected_features:
            anomalies = run_anomaly_detection(dataset, selected_features, contamination)
        else:
            st.sidebar.info("최소 1개의 수치형 컬럼을 선택하세요.")
            contamination = 0.05

    contamination_value = locals().get("contamination", 0.05)
    dataset_summary = build_dataset_summary(
        dataset, anomalies, selected_features, contamination_value
    )
    st.session_state["dataset_summary"] = dataset_summary

    st.subheader("이상 탐지 결과")
    if anomalies is None or anomalies.empty:
        st.write("이상치가 발견되지 않았습니다.")
    else:
        total_anomalies = int(anomalies["anomaly"].sum())
        st.metric("Detected anomalies", f"{total_anomalies}")
        st.dataframe(
            dataset.loc[anomalies[anomalies["anomaly"]].index].assign(
                anomaly_score=anomalies.loc[anomalies["anomaly"], "score"]
            )
        )

    st.subheader("시각화")
    if not numeric_columns:
        st.info("시각화를 위해 필요한 수치형 컬럼이 없습니다.")
    else:
        metric_column = st.selectbox(
            "시각화할 수치형 컬럼",
            options=numeric_columns,
            index=0,
        )
        time_candidates = [
            col
            for col in dataset.columns
            if np.issubdtype(dataset[col].dtype, np.datetime64)
            or "time" in col.lower()
            or "date" in col.lower()
        ]
        time_column = None
        if time_candidates:
            time_column = st.selectbox(
                "시간 축으로 사용할 컬럼",
                options=["행 인덱스 사용"] + time_candidates,
            )
            if time_column == "행 인덱스 사용":
                time_column = None
        render_anomaly_visualisation(dataset, anomalies, time_column, metric_column)

    st.divider()
    st.header("챗봇과 대화하기")
    chat_model = prepare_chat_model()
    if chat_model is None:
        st.warning(
            "GOOGLE_API_KEY 환경 변수가 설정되지 않아 챗봇 응답을 생성할 수 없습니다."
        )

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input("데이터 분석과 관련된 질문을 입력하세요.")
    if user_question:
        st.session_state["chat_history"].append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        if chat_model is None:
            assistant_reply = (
                "현재 LLM API 키가 없어 자동 응답을 제공할 수 없습니다. "
                "환경 변수 GOOGLE_API_KEY를 설정한 뒤 새로고침 해주세요."
            )
        else:
            context_payload = BASE_SYSTEM_PROMPT.format(
                context=dataset_summary
            )
            messages: List[SystemMessage | HumanMessage | AIMessage] = [
                SystemMessage(content=context_payload)
            ]
            for item in st.session_state["chat_history"]:
                if item["role"] == "user":
                    messages.append(HumanMessage(content=item["content"]))
                elif item["role"] == "assistant":
                    messages.append(AIMessage(content=item["content"]))

            response = chat_model.invoke(messages)
            assistant_reply = response.content

        st.session_state["chat_history"].append(
            {"role": "assistant", "content": assistant_reply}
        )
        with st.chat_message("assistant"):
            st.markdown(assistant_reply)


if __name__ == "__main__":
    main()
