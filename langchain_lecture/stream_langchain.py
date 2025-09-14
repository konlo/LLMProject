import os

import openai
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

# 환경 변수 로드
load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")

# OpenAI API Key 설정
openai.api_key = api_key

# Streamlit 페이지 설정
st.set_page_config(page_title="System Prompt for Information", page_icon=":robot_face:")

st.title("Chatting based on System Prompt")
st.markdown(
    """
    이번 프로젝트에서는 LangChain을 활용해서 System Prompt와 PDF 파일을 활용하여 답변을 생성하는 챗봇을 구현해보겠습니다.
    앞서 여러분이 사용하신 System Prompt 기반 챗봇과 본질적으로 유사하며, LangChain을 활용하여 더욱 다양한 기능을 추가할 수 있습니다.
    """
)

# 대화 저장소 및 context 초기화
if "chat_history_5" not in st.session_state:
    st.session_state["chat_history_5"] = []

if "system_prompt_5" not in st.session_state:
    st.session_state["system_prompt_5"] = ""

store = {}

llm = ChatOpenAI(model="gpt-4o")
# Load and split PDF document
with st.spinner("Loading PDF..."):
    loader = PyPDFLoader("../Maximizing Muscle Hypertrophy.pdf")
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(pages)
# Create vector store and retriever
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()


def save_message(message, role):
    st.session_state["chat_history_5"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def load_previous_chat():
    for message in st.session_state["chat_history_5"]:
        send_message(message["message"], message["role"], save=False)


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# context 입력 처리
if st.session_state["system_prompt_5"] == "":
    st.write("먼저 시스템 프롬프트를 설정하세요.")
    system_prompt_input = st.text_area(
        "시스템 프롬프트를 입력하세요.",
        height=150,
        placeholder="논문 리뷰 전문가 입니다. 사용자의 질문에 답하세요.",
    )

    if st.button("context 설정"):
        if system_prompt_input:
            st.session_state["system_prompt_5"] = system_prompt_input
            st.success("시스템 프롬프트가 설정되었습니다. 이제 대화를 시작하세요.")
            st.experimental_rerun()  # 페이지를 리로드하여 채팅 화면으로 전환
        else:
            st.warning("시스템 프롬프트를 입력하세요.")
else:
    st.success("시스템 프롬프트가 설정되었습니다. 대화를 시작해주세요.")

    ####################################################################################################
    # [TODO 1] 아래 프롬프트 템플릿은 Hard-coded System Prompt를 사용합니다.
    # 이를 사용자가 입력한 System Prompt를 사용하도록 수정하세요.
    system_prompt = st.session_state["system_prompt_5"] + "\n{context}"
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    ####################################################################################################
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # [TODO 2] RunnableWithMessageHistory 클래스를 사용하여 대화형 RAG 체인을 생성하세요.
    conversational_rag_chain =RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key = "hist",
        output_messages_key="answer",
        )
    # 기존 대화 불러오기
    load_previous_chat()

    # 사용자 입력 처리
    user_input = st.chat_input("질문을 입력하세요.")
    if user_input:
        send_message(user_input, "user")

        if openai.api_key is None:
            send_message(
                "API 키가 설정되지 않았습니다. 환경 변수를 확인하세요.",
                "system",
                save=False,
            )
        else:
            try:
                response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": "paperbot"}},
                )
                print(response)
                response =response["answer"]
                send_message(response, "assistant")
            except Exception as e:
                send_message(f"오류 발생: {e}", "system", save=False)
