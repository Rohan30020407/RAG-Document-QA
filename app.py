import streamlit as st
st.title("🚀 RAG App Starting...")
st.write("Streamlit UI load ho gayi.")
import os
import openai
from dotenv import load_dotenv

# -----------------------------
# LangChain Imports (NEWSTYLE)
# -----------------------------

# LLM
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings

# Text Splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Document Loader
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Vector Store
from langchain_community.vectorstores import FAISS

# Prompts
from langchain_core.prompts import ChatPromptTemplate

# New LangChain "runnables" (replacement for chains)
from langchain_core.runnables import RunnableParallel, RunnablePassthrough


# -----------------------------
# ENV Setup
# -----------------------------
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "RAG-QA-GROQ"


# -----------------------------
# Initialize Groq LLM
# -----------------------------
# llm = ChatGroq(
#     groq_api_key=os.getenv("GROQ_API_KEY"),
#     model_name="llama-3.1-8b-instant",
# )

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-120b",
)

# -----------------------------
# Prompt Template
# -----------------------------
prompt = ChatPromptTemplate.from_template("""
Answer the question using ONLY the context given below.

<context>
{context}
</context>

Question: {question}
""")


# -----------------------------
# Vector Embedding Builder
# -----------------------------
def create_vector_embedding():
    if "vectors" not in st.session_state:

        folder_path = "E:/Generative_AI_With_Python/1-Basics+Of+Langchain/Project_1_RAG/Research_Papers"

        # Load PDFs
        st.session_state.loader = PyPDFDirectoryLoader(folder_path)
        st.session_state.docs = st.session_state.loader.load()

        st.write("📄 Total documents loaded:", len(st.session_state.docs))

        if len(st.session_state.docs) == 0:
            st.error("❌ No PDF documents found. Check the folder path.")
            return

        # Quick sample text check
        st.write("🔍 Sample Extracted Text:")
        st.write(st.session_state.docs[0].page_content[:300])

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        st.session_state.final_documents = splitter.split_documents(
            st.session_state.docs
        )

        st.write("📚 Total Chunks Created:", len(st.session_state.final_documents))

        if len(st.session_state.final_documents) == 0:
            st.error("❌ Unable to extract text. PDFs may be scanned images or encrypted.")
            return

        # Embeddings
        embeddings = OpenAIEmbeddings()

        # Build FAISS vector DB
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents,
            embeddings
        )

        st.success("✅ Vector Database Ready!")


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📘 RAG Document Q&A — Groq + GPT-OSS")

user_query = st.text_input("Ask a question from your research papers:")

if st.button("Create Embeddings"):
    create_vector_embedding()


# -----------------------------
# RAG Query Execution
# -----------------------------
if user_query:

    if "vectors" not in st.session_state:
        st.error("Embedding not ready! Click 'Create Embeddings' first.")
        st.stop()

    retriever = st.session_state.vectors.as_retriever()

    # Build RAG pipeline using new LangChain Runnables
    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )

    response = rag_chain.invoke(user_query)

    st.subheader("🧠 Answer")
    st.write(response.content)

    # Show retrieved documents
    with st.expander("📄 Relevant Document Chunks"):
        docs = retriever.invoke(user_query)
        for i, doc in enumerate(docs):
            st.write(f"### Chunk {i+1}")
            st.write(doc.page_content)
            st.markdown("---")
