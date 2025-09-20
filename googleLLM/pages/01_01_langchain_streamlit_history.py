import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# LangChain + Gemini
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

# 콜백 수집용
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks import StdOutCallbackHandler

import matplotlib.pyplot as plt


# -----------------------------
# 콜백: 모든 이벤트(프롬프트/툴 I/O/액션)를 수집
# -----------------------------
class CollectAllCallback(BaseCallbackHandler):
    def __init__(self):
        self.events = []

    # LLM 프롬프트(원문)
    def on_llm_start(self, serialized, prompts, **kwargs):
        self.events.append({"type": "llm_start", "prompts": prompts})

    # LLM 응답 텍스트(generations)
    def on_llm_end(self, response, **kwargs):
        gens = [[g.text for g in gen] for gen in response.generations]
        self.events.append({"type": "llm_end", "generations": gens})

    # 툴 시작/종료 (예: python_repl_ast 코드 실행)
    def on_tool_start(self, serialized, input_str, **kwargs):
        self.events.append({
            "type": "tool_start",
            "tool": serialized.get("name"),
            "input": input_str,
        })

    def on_tool_end(self, output, **kwargs):
        self.events.append({
            "type": "tool_end",
            "output": output,
        })

    # 에이전트 고수준 액션/종료
    def on_agent_action(self, action, **kwargs):
        self.events.append({
            "type": "agent_action",
            "tool": action.tool,
            "tool_input": action.tool_input,
            "log": action.log,
        })

    def on_agent_finish(self, finish, **kwargs):
        self.events.append({
            "type": "agent_finish",
            "return_values": finish.return_values,
        })


# =============================
# App 시작
# =============================
load_dotenv()  # .env 로드

st.set_page_config(page_title="DF Chatbot (Gemini)", page_icon="✨", layout="wide")
st.title("✨ DataFrame Chatbot (Gemini + LangChain)")
st.caption("Uses Google Gemini through LangChain's `create_pandas_dataframe_agent`.")

# 데이터 로드
df_ride_booking = pd.read_csv(
    # "https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv"
    "/Users/najongseong/dataset/ncr_ride_bookings.csv"
)

# 환경변수 키 체크
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY가 설정되어 있지 않습니다. .env 또는 환경변수를 확인하세요.")
    st.stop()

# LLM 세팅
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=GOOGLE_API_KEY,     # ADC 미사용 환경 대비
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    streaming=True,             # 진행상황 스트리밍
)

# 미리보기
st.subheader("Preview")
st.dataframe(df_ride_booking.head(10), use_container_width=True)

# 에이전트 생성 (주의: 여기서는 return_intermediate_steps를 넣지 않는다)
agent = create_pandas_dataframe_agent(
    llm=llm,
    df=df_ride_booking,
    verbose=True,
    allow_dangerous_code=True,
    agent_executor_kwargs={
        "handle_parsing_errors": True,   # ✅ 가능
        # "return_intermediate_steps": True,  ❌ 넣으면 TypeError 가능
    },
)

# 생성 후 런타임에 속성으로 활성화 (중복 전달 에러 회피)
# (LangChain의 AgentExecutor는 이 속성을 갖고 있으며 런타임에 변경 가능)
try:
    agent.return_intermediate_steps = True
except Exception:
    pass

st.write("---")
user_q = st.chat_input("Ask a question about your data (예: '상위 5개 항목의 TBW를 보여줘')")

if user_q:
    # 좌/우 컬럼: 좌측은 실시간 로그, 우측은 결과
    left, right = st.columns([1, 1])
    with left:
        st.subheader("실시간 실행 로그")
        st_cb = StreamlitCallbackHandler(st.container())  # 단계별 진행 UI

    # 커스텀 콜백 수집기
    collector = CollectAllCallback()

    # Python 툴 힌트
    TOOL_HINT = (
        "When you use the python tool, return ONLY raw Python code "
        "(no backticks, no 'Action:' or 'Output:' lines, no prose). "
        "Put each statement on its own line. If you use 'pd', add 'import pandas as pd'. "
        "For charts, import matplotlib.pyplot as plt and end with 'plt.show()'."
    )

    with st.spinner("Thinking with Gemini..."):
        # invoke 시 return_intermediate_steps를 config로 넘기지 않는다
        # (상단에서 agent.return_intermediate_steps = True로 설정)
        result = agent.invoke(
            {"input": f"{TOOL_HINT}\n\n{user_q}"},
            {"callbacks": [st_cb, collector, StdOutCallbackHandler()]},
        )

    st.success("Done.")

    # 최종 답변
    final = result.get("output", result)
    with right:
        st.subheader("Answer")
        st.write(final)

        # matplotlib figure 자동 수집/표시
        figs = [plt.figure(n) for n in plt.get_fignums()]
        for f in figs:
            st.pyplot(f, use_container_width=True)
        plt.close("all")

    # =============================
    # 중간 스텝 & 이벤트 타임라인 표시
    # =============================
    st.write("---")
    st.subheader("intermediate_steps (툴 실행 상세)")

    # result["intermediate_steps"]는 (AgentAction, observation) 튜플들의 리스트일 가능성
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

    # 수집된 콜백 이벤트들을 순서대로 출력
    for j, e in enumerate(collector.events, 1):
        with st.expander(f"Event {j}: {e.get('type')}"):
            # LLM 프롬프트는 길 수 있으니 코드 블록으로
            if e.get("type") == "llm_start":
                prompts = e.get("prompts", [])
                for k, p in enumerate(prompts, 1):
                    st.markdown(f"**Prompt {k}**")
                    st.code(p)
            else:
                st.write(e)
