import os
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

PROJECT_NAME = "DocuMind"

st.set_page_config(page_title=f"{PROJECT_NAME} | Chat with your PDF", page_icon="📄")
st.title(f"📄 {PROJECT_NAME}")
st.caption("Upload a PDF and chat with it using AI.")

# ---------- Session State ----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (role, message)

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# ---------- Upload PDF (main area, always visible) ----------
upload_col, clear_col = st.columns([4, 1])

with upload_col:
    uploaded_file = st.file_uploader("📤 Upload your PDF here", type=["pdf"])

with clear_col:
    st.write("")  # spacing to align button with uploader
    st.write("")
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

if uploaded_file is not None and uploaded_file.name != st.session_state.pdf_name:
    with st.spinner("Processing PDF..."):
        # Save uploaded file to a temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        # Load and split PDF
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(docs)

        # Embeddings + Vectorstore (in-memory, no persistence needed per upload)
        embedding_model = MistralAIEmbeddings(model="mistral-embed")

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model
        )

        st.session_state.retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
        )
        st.session_state.pdf_name = uploaded_file.name
        st.session_state.chat_history = []  # reset chat on new PDF

        os.remove(tmp_path)

    st.success(f"'{uploaded_file.name}' processed successfully!")

if st.session_state.pdf_name:
    st.info(f"📌 Currently chatting with: **{st.session_state.pdf_name}**")

st.divider()

# ---------- LLM + Prompt ----------
llm = ChatMistralAI(model="mistral-small-latest")

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
query = st.chat_input("Ask something about the PDF...")

if query:
    if st.session_state.retriever is None:
        st.warning("Please upload a PDF first.")
    else:
        st.session_state.chat_history.append(("user", query))
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                docs = st.session_state.retriever.invoke(query)
                context = "\n\n".join(doc.page_content for doc in docs)

                messages = prompt.format_messages(
                    context=context,
                    question=query
                )

                response = llm.invoke(messages)
                st.markdown(response.content)

        st.session_state.chat_history.append(("assistant", response.content))
