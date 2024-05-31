import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage, #SystemMessage,  # システムメッセージ
    HumanMessage,#HumanMessage,  # 人間の質問
    AIMessage #AIMessage  # ChatGPTの返答
)
from langchain.callbacks import get_openai_callback

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import YoutubeLoader

def init_select_mode():
    st.set_page_config(
        page_title="ChatGPTを流用したwebサイト",
        page_icon="🤗"
    )
    mode = st.sidebar.radio("使う機能を選択:", ("gpt_chat", "web_summary","youtube_summary"))
    if mode == "gpt_chat":
        st.header("ChatGPT機能 🤗")
    elif mode == "web_summary":
        st.header("webサイトの要約 🤗")
    else:
        st.header("Youtubeの要約 🤗")
    st.sidebar.title("Options")
    mode_name = mode
    return mode_name

def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="システムメッセージ：あなたの手助けを行います")
        ]
        st.session_state.costs = []

def select_model():
    model = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
    if model == "GPT-3.5":
        model_name = "gpt-3.5-turbo"
    else:
        model_name = "gpt-4"

    # スライダーを追加し、temperatureを0から2までの範囲で選択可能にする
    # 初期値は0.0、刻み幅は0.01とする
    temperature = st.sidebar.slider("Temperature:", min_value=0.0, max_value=2.0, value=0.0, step=0.01)

    return ChatOpenAI(temperature=temperature, model_name=model_name)

def get_url_input():
    url = st.text_input("URL: ", key="input")
    return url

def get_youtube_url_input():
    url = st.text_input("Youtube URL: ", key="input")
    return url

def validate_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def get_content(url):
    try:
        with st.spinner("Fetching Content ..."):
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            # fetch text from main (change the below code to filter page)
            if soup.main:
                return soup.main.get_text()
            elif soup.article:
                return soup.article.get_text()
            else:
                return soup.body.get_text()
    except:
        st.write('something wrong')
        return None

def get_youtube_document(url):
    with st.spinner("Fetching Content ..."):
        loader = YoutubeLoader.from_youtube_url(
            url,
            add_video_info=True,  # タイトルや再生数も取得できる
            language=['en', 'ja']  # 英語→日本語の優先順位で字幕を取得
        )
        return loader.load()

def summarize(llm, docs):
    prompt_template = """Write a concise Japanese summary of the following transcript of Youtube Video.
        ============
        {text}
        ============

        ここから日本語で書いてね
        必ず10段落以内の600文字以内で簡潔にまとめること:
        """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

    with get_openai_callback() as cb:
        chain = load_summarize_chain( 
            llm,
            chain_type="stuff",
            verbose=True,
            prompt=PROMPT
        )
        response = chain({"input_documents": docs}, return_only_outputs=True)
        
    return response['output_text'], cb.total_cost

def build_prompt(content, n_chars=300):
    return f"""以下はとある。Webページのコンテンツである。内容を{n_chars}程度でわかりやすく要約してください。
            ========
            {content[:1000]}
            ========
            日本語で書いてね！
            """

def get_answer(llm, messages):
    with get_openai_callback() as cb:
        answer = llm(messages)
    return answer.content, cb.total_cost

def main():
    mode_name = init_select_mode()

    llm = select_model()
    init_messages()

    if mode_name == "gpt_chat":
        if user_input := st.chat_input("聞きたいことを入力してね！"):
            st.session_state.messages.append(HumanMessage(content=user_input))
            with st.spinner("ChatGPT is typing ..."):
                answer, cost = get_answer(llm, st.session_state.messages)
            st.session_state.messages.append(AIMessage(content=answer))
            st.session_state.costs.append(cost)
        messages = st.session_state.get('messages', [])
        for message in messages:
            if isinstance(message, AIMessage):
                with st.chat_message('assistant'):
                    st.markdown(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message('user'):
                    st.markdown(message.content)
            else:  # isinstance(message, SystemMessage):
                st.write(f"System message: {message.content}")
    elif mode_name == "web_summary":
        container = st.container()
        response_container = st.container()

        with container:
            url = get_url_input()
            is_valid_url = validate_url(url)
            if not is_valid_url:
                st.write('Please input valid url')
                answer = None
            else:
                # URLのパース
                content = get_content(url)
                if content:
                    # パースした内容と質問文書を作成
                    prompt = build_prompt(content)
                    st.session_state.messages.append(HumanMessage(content=prompt))
                    with st.spinner("ChatGPT is typing ..."):
                        answer, cost = get_answer(llm, st.session_state.messages)
                    st.session_state.costs.append(cost)
                else:
                    answer = None
        if answer:
            with response_container:
                st.markdown("## Summary")
                st.write(answer)
                st.markdown("---")
                st.markdown("## Original Text")
                st.write(content)
    else:
        container = st.container()
        response_container = st.container()

        with container:
            url = get_youtube_url_input()
            if url:
                document = get_youtube_document(url)
                with st.spinner("ChatGPT is typing ..."):
                    output_text, cost = summarize(llm, document)
                st.session_state.costs.append(cost)
            else:
                output_text = None

        if output_text:
            with response_container:
                st.markdown("## Summary")
                st.write(output_text)
                st.markdown("---")
                st.markdown("## Original Text")
                st.write(document)

    costs = st.session_state.get('costs', [])
    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")

if __name__ == '__main__':
    main()