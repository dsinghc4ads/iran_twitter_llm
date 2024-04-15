import streamlit as st
import re
import requests
from dotenv import load_dotenv
import logging


logging.basicConfig(level=logging.INFO)
load_dotenv(".env")


st.set_page_config(
    page_title="Iran Twitter Sources RAG",
    page_icon="🦙",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)
st.title("Chat with Iranian Twitter Sources, powered by LlamaIndex 💬🇮🇷")
st.info(
    "Project Documentation in [C4ADS Notion Board](https://www.notion.so/Retrieval-Augmented-Generation-RAG-cda699b8b6d04581b6f254a5a94bd7a3)",
    icon="📃",
)

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me about Iranian Twitter sources"}
    ]


def query_api(question):
    url = "http://ingest_service:8080/query/"
    payload = {"question": question}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "Failed to get response from API"}


def parse_response(response):
    response_text = response["response"]
    pattern = r"> Source \(Doc id: [a-f0-9-]{36}\): "
    relevant_sources = re.split(pattern, response.get("relevant_sources", ""))
    relevant_sources = [source.strip() for source in relevant_sources if source != ""]
    return response_text, relevant_sources


if prompt := st.chat_input(
    "Your question"
):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = query_api(prompt).get('response')
            if "error" in response:
                st.error(response["error"])
            else:
                response_text = response.get('response')
                source_nodes = response.get('source_nodes')
                st.write(response_text)
                message = {"role": "assistant", "content": response_text}
                st.session_state.messages.append(message)
                if source_nodes:
                    for i, node in enumerate(source_nodes, start=1):
                        node = node.get('node')
                        document_name = f"{node['text']}"#['metadata']['subject']}\nTO: {node['metadata']['to']}\nFROM: {node['metadata']['from']}"
                        placeholder = st.empty()
                        with placeholder.container():
                            with st.expander(f"{document_name}"):
                                st.text(node.get('text'))
