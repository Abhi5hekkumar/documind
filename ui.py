import os
import sys
import tempfile

import streamlit as st
from dotenv import load_dotenv

# Fix SQLite compatibility for Chroma
import pysqlite3

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate


# ---------- Environment ----------
load_dotenv()


# ---------- Project ----------
PROJECT_NAME = "DocuMind"

st.set_page_config(
    page_title=f"{PROJECT_NAME} | Chat with your PDF",
    page_icon="📄"
)

st.title(f"📄 {PROJECT_NAME}")
st.caption("Upload a PDF and chat with it using AI.")


# ---------- Session State ----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None


# ---------- Mistral API Key ----------
# Streamlit Cloud:
# Settings → Secrets
#
# Add:
# MISTRAL_API_KEY = "your-api-key"

try:
    MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]
except Exception:
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    st.error(
        "MISTRAL_API_KEY is not configured. "
        "Add it to Streamlit Secrets or your environment variables."
    )
    st.stop()


# ---------- Upload PDF ----------
upload_col, clear_col = st.columns([4, 1])

with upload_col:
    uploaded_file = st.file_uploader(
        "📤 Upload your PDF here",
        type=["pdf"]
    )

with clear_col:
    st.write("")
    st.write("")

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()


# ---------- Process PDF ----------
if (
    uploaded_file is not None
    and uploaded_file.name != st.session_state.pdf_name
):

    with st.spinner("Processing PDF..."):

        # Save uploaded PDF temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp_file:

            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        try:
            # Load PDF
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

            # Split PDF into chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(docs)

            # Mistral embeddings
            embedding_model = MistralAIEmbeddings(
                model="mistral-embed",
                api_key=MISTRAL_API_KEY
            )

            # Create vector store
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model
            )

            # Create retriever
            st.session_state.retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 4,
                    "fetch_k": 10,
                    "lambda_mult": 0.5
                }
            )

            st.session_state.pdf_name = uploaded_file.name
            st.session_state.chat_history = []

        finally:
            # Delete temporary PDF
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    st.success(
        f"'{uploaded_file.name}' processed successfully!"
    )


# ---------- Current PDF ----------
if st.session_state.pdf_name:
    st.info(
        f"📌 Currently chatting with: "
        f"**{st.session_state.pdf_name}**"
    )


st.divider()


# ---------- LLM ----------
llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0,
    api_key=MISTRAL_API_KEY
)


# ---------- Prompt ----------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context, say:

"I could not find the answer in the document."
"""
        ),
        (
            "human",
            """
Context:
{context}

Question:
{question}
"""
        )
    ]
)


# ---------- Display Chat History ----------
for role, message in st.session_state.chat_history:

    with st.chat_message(role):
        st.markdown(message)


# ---------- Chat Input ----------
query = st.chat_input(
    "Ask something about the PDF..."
)


if query:

    # Make sure a PDF has been uploaded
    if st.session_state.retriever is None:

        st.warning(
            "Please upload a PDF first."
        )

    else:

        # Add user message
        st.session_state.chat_history.append(
            ("user", query)
        )

        with st.chat_message("user"):
            st.markdown(query)


        # Assistant response
        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                try:

                    # Retrieve relevant document chunks
                    docs = st.session_state.retriever.invoke(
                        query
                    )

                    # Combine retrieved chunks
                    context = "\n\n".join(
                        doc.page_content
                        for doc in docs
                    )

                    # Create prompt messages
                    messages = prompt.format_messages(
                        context=context,
                        question=query
                    )

                    # Call Mistral
                    import time 
                    time.sleep(1)
                    response = llm.invoke(messages)

                    # Display answer
                    st.markdown(response.content)

                    # Save answer to chat history
                    st.session_state.chat_history.append(
                        ("assistant", response.content)
                    )

                except Exception as e:

                    st.error(
                        f"Error type: {type(e).__name__}"
                    )

                    st.error(str(e))

                    st.exception(e)
