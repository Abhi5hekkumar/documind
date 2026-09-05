import os
import sys
import tempfile
import time

import streamlit as st
from dotenv import load_dotenv

# ============================================================
# FIX SQLITE COMPATIBILITY FOR CHROMA
# ============================================================

import pysqlite3

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")


# ============================================================
# LANGCHAIN IMPORTS
# ============================================================

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# PROJECT
# ============================================================

PROJECT_NAME = "DocuMind"

st.set_page_config(
    page_title=f"{PROJECT_NAME} | Chat with your PDF",
    page_icon="📄"
)

st.title(f"📄 {PROJECT_NAME}")
st.caption("Upload a PDF and chat with it using AI.")


# ============================================================
# SESSION STATE
# ============================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

if "last_llm_call" not in st.session_state:
    st.session_state.last_llm_call = 0.0


# ============================================================
# MISTRAL API KEY
# ============================================================

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


# ============================================================
# MODELS
# ============================================================

@st.cache_resource
def load_embedding_model():
    """
    Local embedding model.

    This does NOT use the Mistral API.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def load_llm(api_key):
    """
    Mistral is used only for generating the final answer.
    """
    return ChatMistralAI(
        model="mistral-small-latest",
        temperature=0,
        api_key=api_key
    )


embedding_model = load_embedding_model()
llm = load_llm(MISTRAL_API_KEY)


# ============================================================
# UPLOAD PDF
# ============================================================

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


# ============================================================
# PROCESS PDF
# ============================================================

if (
    uploaded_file is not None
    and uploaded_file.name != st.session_state.pdf_name
):

    with st.spinner("Processing PDF..."):

        # ----------------------------------------------------
        # Save uploaded PDF temporarily
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp_file:

            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        try:

            # ------------------------------------------------
            # Load PDF
            # ------------------------------------------------

            loader = PyPDFLoader(tmp_path)
            docs = loader.load()


            # ------------------------------------------------
            # Split PDF into chunks
            # ------------------------------------------------

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(docs)


            # ------------------------------------------------
            # LOCAL EMBEDDINGS
            # ------------------------------------------------
            #
            # No Mistral API call happens here.
            #

            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model
            )


            # ------------------------------------------------
            # Create retriever
            # ------------------------------------------------

            st.session_state.retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 4,
                    "fetch_k": 10,
                    "lambda_mult": 0.5
                }
            )


            # ------------------------------------------------
            # Save PDF information
            # ------------------------------------------------

            st.session_state.pdf_name = uploaded_file.name
            st.session_state.chat_history = []


        finally:

            # ------------------------------------------------
            # Delete temporary PDF
            # ------------------------------------------------

            if os.path.exists(tmp_path):
                os.remove(tmp_path)


    st.success(
        f"'{uploaded_file.name}' processed successfully!"
    )


# ============================================================
# CURRENT PDF
# ============================================================

if st.session_state.pdf_name:

    st.info(
        f"📌 Currently chatting with: "
        f"**{st.session_state.pdf_name}**"
    )


st.divider()


# ============================================================
# PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context, say:

"I could not find the answer in the document."

Do not make up information.
Do not use outside knowledge.
Keep the answer clear and relevant.
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


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for role, message in st.session_state.chat_history:

    with st.chat_message(role):
        st.markdown(message)


# ============================================================
# CHAT INPUT
# ============================================================

query = st.chat_input(
    "Ask something about the PDF..."
)


if query:

    # --------------------------------------------------------
    # Make sure a PDF has been uploaded
    # --------------------------------------------------------

    if st.session_state.retriever is None:

        st.warning(
            "Please upload a PDF first."
        )

    else:

        # ----------------------------------------------------
        # Add user message to history
        # ----------------------------------------------------

        st.session_state.chat_history.append(
            ("user", query)
        )

        with st.chat_message("user"):
            st.markdown(query)


        # ----------------------------------------------------
        # Assistant response
        # ----------------------------------------------------

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                try:

                    # ========================================
                    # STEP 1: RETRIEVE DOCUMENT CHUNKS
                    # ========================================

                    docs = st.session_state.retriever.invoke(
                        query
                    )


                    # ========================================
                    # STEP 2: COMBINE CONTEXT
                    # ========================================

                    context = "\n\n".join(
                        doc.page_content
                        for doc in docs
                    )


                    # ========================================
                    # STEP 3: CREATE PROMPT
                    # ========================================

                    messages = prompt.format_messages(
                        context=context,
                        question=query
                    )


                    # ========================================
                    # STEP 4: RATE-LIMIT PROTECTION
                    # ========================================

                    current_time = time.time()

                    elapsed = (
                        current_time
                        - st.session_state.last_llm_call
                    )

                    if elapsed < 1.0:

                        time.sleep(
                            1.0 - elapsed
                        )


                    # ========================================
                    # STEP 5: CALL MISTRAL
                    # ========================================

                    response = llm.invoke(messages)

                    st.session_state.last_llm_call = time.time()


                    # ========================================
                    # STEP 6: DISPLAY ANSWER
                    # ========================================

                    st.markdown(
                        response.content
                    )


                    # ========================================
                    # STEP 7: SAVE ANSWER
                    # ========================================

                    st.session_state.chat_history.append(
                        (
                            "assistant",
                            response.content
                        )
                    )


                # ============================================
                # ERROR HANDLING
                # ============================================

                except Exception as e:

                    error_message = str(e)

                    if (
                        "429" in error_message
                        or "rate limit" in error_message.lower()
                    ):

                        st.warning(
                            "⏳ Mistral API rate limit reached. "
                            "Please wait a little while and try again."
                        )

                    else:

                        st.error(
                            f"❌ {type(e).__name__}"
                        )

                        st.error(
                            error_message
                        )

                        # Show traceback while developing
                        st.exception(e)
