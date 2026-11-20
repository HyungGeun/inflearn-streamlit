from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from config import answer_examples

# LangChain이 LLM에 전달할 대화 기록입니다. session_id별로 프로세스 메모리에 저장합니다.
store = {}

# session_id: str은 문자열을 받는다는 타입 힌트입니다.
# -> BaseChatMessageHistory는 대화 기록 객체를 반환한다는 타입 힌트입니다.
# 예를 들어 def add(a: int, b: int) -> int: 에서 -> int는 반환값이 정수라는 뜻입니다.
# 타입 힌트는 값을 변환하거나 검사하지 않습니다. 실제로 돌려주는 값은 return이 정합니다.
# BaseChatMessageHistory는 기록 조회·추가 등 공통 기능을 정의하는 클래스입니다.
# 이 함수가 실제로 만들고 반환하는 객체는 메모리에 기록을 저장하는 InMemoryChatMessageHistory입니다.
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        # 처음 보는 세션 ID라면 빈 기록을 만든 뒤, 이후 호출에서 같은 기록을 꺼냅니다.
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


def get_retriever():
    # 질문은 벡터로 변환하고, Pinecone의 기존 문서 벡터와 유사도를 비교합니다.
    embedding = OpenAIEmbeddings(model='text-embedding-3-large')
    index_name = "tax-index-markdown"
    database = PineconeVectorStore(
        embedding=embedding,
        index_name=index_name
    )
    # 검색 결과로 유사한 문서를 최대 4개 가져오도록 설정합니다.
    retriever = database.as_retriever(search_kwargs={"k":4}) # 질문과 유사한 문서 검색
    return retriever


def get_history_retriever():
    llm=get_llm()
    retriever=get_retriever()

    # ================= {context} 에 필요한 정보 가져오기 시작 ======================

    # 내용. 챗 히스토리와 input을 줄테니, 히스토리 내용 흐름에 따라서 input을 새로 만들어줘
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the history, "
        "formulate a standalone question which cand be understand "
        "without the chat history. Do not answer the question. Return only the rewritten search query."
        "just reformulate it if needed and otherwise return it as is."
    )

    # chat_history 자리에는 이전 메시지 목록을, input 자리에는 이번 질문을 넣습니다.
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # 입력 질문과 chat_history로 새로운 질문을 만들어 retriever 함. 리턴 문서목록
    # create_history_aware_retriever => 검색한 문서 목록({context} 로 추후 사용)

    # 실행할 때 input(이번 질문)과 chat_history(이전 대화)를 확인합니다.
    # 기록이 비어 있으면 input을 그대로 retriever에 넘겨 Pinecone에서 검색합니다.

    # 기록이 있으면 contextualize_q_prompt에 질문과 기록을 넣어 llm에 보냅니다.
    # 🌟 Why? 👉 임베딩에 필요한 질문을 새롭게 가공하는건 llm 이 해줘야 하기 때문에.

    # 예: 앞에서 소득세를 이야기했다면 "그럼 신고 기한은?"을
    # "소득세 신고 기한은?"처럼 바꾼 뒤 그 문장으로 검색합니다.

    # 이 단계가 다음 단계에 넘기는 값은 검색 질문이나 답변이 아니라 검색된 문서 목록입니다.
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    return history_aware_retriever
    # ================= {context} 에 필요한 정보 가져오기 종료 ======================


def get_llm(model='gpt-4o'):
    # 질문 다듬기와 최종 답변 생성에 사용할 채팅 모델 객체를 만듭니다.
    llm = ChatOpenAI(
        model=model
    )
    return llm


def get_dictionary_chain():
    dictionary = ["사람을 나타내는 표현 -> 거주자"]
    
    # f-string은 사전을 먼저 삽입합니다. 이중 중괄호는 LangChain 입력 자리인 {question}을 남깁니다.
    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        질문을 변경하지 않는다면, 질문만 리턴해주세요.
        사전: {dictionary}

        질문: {{question}}
    """)

    # | 는 앞 단계의 출력을 다음에 전달합니다. 프롬프트 → 모델 → 문자열 순서입니다.
    dictionary_chain = prompt | get_llm() | StrOutputParser()
    return dictionary_chain


def get_rag_chain():
    llm=get_llm()
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{answer}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=answer_examples,
    )

    history_aware_retriever=get_history_retriever()
    
    system_prompt = (
        "당신은 소득세법 전문가입니다. 사용자의 소득세법에 관한 질문에 답변해주세요"
        "아래에 제공된 문서를 활용해서 답변해주시고"
        "답변을 알 수 없다면 모른다고 답변해주세요"
        "답변을 제공할 때는 소득세법 (XX조)에 따르면 이라고 시작하면서 답변해주시고"
        "2-3 문장 정도의 짧은 내용의 답변을 원합니다"
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )
    # stuff는 검색된 문서 내용을 한 프롬프트에 모아 넣는다는 뜻입니다.
    # 🌟 실행할 때 context로 받은 문서 목록에서 본문을 꺼내 빈 줄로 이어 붙입니다.
    # 예: ["문서 A 본문", "문서 B 본문"]의 내용을 한 문자열로 만들어 {context}에 넣습니다.
    # qa_prompt에 문서 본문, 이번 질문(input), 이전 대화(chat_history)를 채워 llm에 보냅니다.
    # 이 단계가 반환하는 값은 모델이 작성한 답변 문자열입니다.
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # 🌟 create_retrieval_chain은 위의 검색 단계와 답변 단계를 순서대로 연결합니다.
    # 이 줄에서 검색이나 답변 생성이 바로 실행되지는 않습니다.
    # 아래 get_ai_response에서 stream으로 이 체인을 사용하면 다음 순서로 실행됩니다.
    # 1. history_aware_retriever로 문서를 검색하고, 💫 입력 딕셔너리에 {context} 키로 추가합니다.
    # 2. question_answer_chain에 그 💫 딕셔너리를 넘기고, 생성된 답변을 💫 answer 키로 추가합니다.
    # 검색용 질문이 재작성되어도 기존 input(사전 단계에서 다듬은 질문)은 그대로 전달됩니다.
    # 결과 예: {"input": "그럼 신고 기한은?", "chat_history": [...],
    #           "context": [문서A, 문서B], "answer": "신고 기한은 ..."}
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # 🌟 RunnableWithMessageHistory: store 변수 에서 이전 메시지를 꺼내 chat_history 에 저장

    # pick은 검색 문서 등 중간 결과에서 answer만 골라 화면으로 보냅니다.
    # 실행 전에는 이전 기록을 주입하고, 실행 후에는 input과 answer를 기록합니다.
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick("answer")

    return conversational_rag_chain


def get_ai_response(user_message):
    dictionary_chain = get_dictionary_chain()
    rag_chain = get_rag_chain()
    # 질문을 사전 규칙으로 다듬고, 그 문자열을 RAG 체인이 읽는 input 키로 전달합니다.
    tax_chain = {'input': dictionary_chain} | rag_chain
    # stream은 답변 조각을 순서대로 받을 수 있는 반복자를 반환합니다.
    ai_response = tax_chain.stream(
        {"question": user_message},
        config={
            # 고정 ID를 사용하므로 같은 서버 프로세스의 접속자들이 이 기록을 공유할 수 있습니다.
            "configurable": {"session_id": "abc123"}
        },
    )

    # print(ai_response)
    return ai_response