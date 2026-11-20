import streamlit as st
# 실행: streamlit run chat.py

from dotenv import load_dotenv
from llm import get_ai_response

# .env를 읽어 OpenAI와 Pinecone 클라이언트가 사용할 환경 변수를 설정합니다.
load_dotenv()

st.set_page_config(page_title="소득세 챗봇", page_icon="🤖")

st.title("🤖 소득세 챗봇")
st.caption("소득세에 관련된 모든것을 답해드립니다!")

# Streamlit은 입력을 제출하면 스크립트를 다시 실행합니다. 화면에 표시할 메시지는 세션에 보관합니다.
if 'message_list' not in st.session_state:
    st.session_state.message_list = []

# 다시 실행될 때 세션에 저장된 메시지를 역할별 말풍선으로 복원합니다.
for message in st.session_state.message_list:
    with st.chat_message(message['role']): # with: 채팅의 메시지 영역(좌우)
        st.write(message['content'])

# := 는 입력을 user_question에 담으면서, 값이 있을 때만 아래 블록을 실행합니다.
if user_question := st.chat_input(placeholder="소득세에 관련된 궁금한 내용들을 말씀해주세요!"):
    with st.chat_message("user"): # "user" 메시지일 경우 해당 컨네이너 안에서 작업
        st.write(user_question) # 입력해라 화면에

    # 원본 질문을 저장해 다음 실행에서도 화면에 표시합니다.
    st.session_state.message_list.append({'role': 'user', 'content': user_question})

    with st.spinner("답변을 생성하는 중입니다"):
        # llm.py가 돌려주는 스트림을 받습니다. 실제 응답 조각은 아래에서 순서대로 읽습니다.
        ai_response = get_ai_response(user_question)
        
        with st.chat_message("ai"): # "ai" 메시지일 경우 해당 컨네이너 안에서 작업
            # 응답 조각을 도착하는 순서대로 표시하고, 합쳐진 답변을 반환값으로 받습니다.
            ai_message = st.write_stream(ai_response)
            st.session_state.message_list.append({'role': 'ai', 'content': ai_message})

