import streamlit as st
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from templates.htmlTemplates import css, bot_template, user_template
from vectorstore_utils import *
from PIL import Image
from langchain.llms import VertexAI
import variables as v
import os
import shutil
from vertexai.language_models import  ChatModel

if "GOOGLE_CLOUD_PROJECT" in os.environ:
     pass
else:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="app/laudechat_sa_credentials.json"

def get_conversation_chain(vectorstore,output_tokens_config):
    """
    Creates and configures a conversational chain using a language model for text-based interactions.

    :param vectorstore: An object representing the vector store for retrieving information.
    :return: A configured conversational chain for text-based interactions.
    """
    llm = VertexAI(
    model_name='text-bison@002',
    max_output_tokens=output_tokens_config,
    temperature=0.1,
    top_p=0.8,top_k=40,
    verbose=True,
    )

    #retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5,"k": 3})
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True,output_key='answer')
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever= retriever ,
        memory=memory,
        return_source_documents=True
    )
    return conversation_chain


def handle_userinput(user_question):
    """
    Handles user input, processes it, and generates a response.

    :param user_question: The user's question or input.
    """
    page_contents=[]
    try:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.response = response['source_documents']
        st.session_state.chat_history = response['chat_history']
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
                st.session_state.last_user_msg = message.content
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
                st.session_state.last_bot_msg = message.content
        st.session_state.my_text = ""
    except TypeError as e:
        mensaje_error = "Click on 'Process' after loading your documents or 'Process bucket'"
        st.warning(mensaje_error)

def reset_session_state():
    keys_to_reset = ["conversation", "chat_history", "last_user_msg", "last_bot_msg", "vectorstore", "documents"]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]

def main():
    """
    Main function for the Laude Chat Streamlit application.
    This function sets up the Streamlit page, handles user input, and processes documents.
    """ 
    st.set_page_config(page_title="Laude Chat",
                       page_icon=":page_facing_up:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
        st.session_state.last_user_msg =None
        st.session_state.last_bot_msg = None
        st.session_state.vectorstore = None
        st.session_state.documents = None

    st.header("Laude Chat :page_facing_up:")
    prompt = st.chat_input("Ask a question about your documents")

    chat_container = st.container()

    if prompt:
        handle_userinput(prompt)
    elif st.session_state.chat_history is not None:
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                chat_container.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                chat_container.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)


    with st.sidebar:
        raw_text_pdf=""
        raw_text_txt=""

        image = Image.open('app/images/laude_logo.jpg')
        st.image(image)

        st.subheader("Chatbot configuration")
        output_tokens_config = st.slider('Maximum Output Length (tokens)', 20, 1024, 256, 
                                         help='100 tokens correspond approximately to 50-70 words in Spanish and 70-90 words in English.')
        v.output_tokens=output_tokens_config
        if st.button("New Session",help="The current conversation will be deleted, and a new session will start."):
            if st.session_state.chat_history is not None:
                del st.session_state["chat_history"]
                del st.session_state["last_user_msg"]
                del st.session_state["last_bot_msg"]
                del st.session_state["vectorstore"]
                del st.session_state["documents"]
                del st.session_state["conversation"]
                st.experimental_rerun() 
        st.subheader("Your documents")
        upload_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)

        if st.button("Process"):
            if not upload_docs:
                st.warning("Please upload at least one document before selecting 'Process'.")
            else:
                with st.spinner("Processing"):
                    st.session_state.documents= upload_docs
                    if os.path.exists("tmp_docs"):
                        shutil.rmtree("tmp_docs")
                    docs=get_docs(upload_docs)
                    embeddings = VertexAIEmbeddings()
                    vectorstore=get_vectorstore_index(docs,embeddings)
                    st.session_state.conversation = get_conversation_chain(vectorstore,v.output_tokens)
                    st.session_state.vectorstore=vectorstore

        if st.button("Process bucket"):
            with st.spinner("Processing"):
                if os.path.exists("tmp_docs"):
                    shutil.rmtree("tmp_docs")
                embeddings = VertexAIEmbeddings()
                vs = check_vectorstore(v.bucket_vs)
                if vs == False:
                    vectorstore=create_vectorstore(v.bucket_name,embeddings)
                else:
                    st.session_state.documents = docs_bucket(v.bucket_vs)
                    vectorstore=load_index_from_gcs(v.bucket_vs,embeddings)
                st.session_state.conversation = get_conversation_chain(
                    vectorstore,v.output_tokens)
                st.session_state.vectorstore=vectorstore

if __name__ == '__main__':
    main()