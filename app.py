import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import tempfile
import os

st.set_page_config(page_title="Chat with PDF", page_icon="📄")
st.title("📄 Chat with your PDF")

# Sidebar for API key
with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("OpenAI API Key", type="password")
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("""
    1. Enter your OpenAI API key
    2. Upload a PDF
    3. Ask questions about it
    """)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# File upload
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file and api_key:
    # Process PDF only once
    if st.session_state.vectorstore is None:
        with st.spinner("Processing PDF..."):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            
            # Load and split PDF
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(documents)
            
            # Create embeddings and vector store
            embeddings = OpenAIEmbeddings(api_key=api_key)
            st.session_state.vectorstore = Chroma.from_documents(chunks, embeddings)
            
            # Clean up temp file
            os.unlink(tmp_path)
            
            st.success(f"PDF processed! {len(chunks)} chunks created.")

    # Chat interface
    if st.session_state.vectorstore:
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
        retriever = st.session_state.vectorstore.as_retriever()
        
        prompt = ChatPromptTemplate.from_template("""
Answer the question based on the context below. If you cannot answer, say "I don't know."

Context:
{context}

Question: {question}

Answer:""")
        
        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        
        # Chat input
        if question := st.chat_input("Ask a question about your PDF"):
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Retrieve relevant docs
                    docs = retriever.invoke(question)
                    context = "\n\n".join([doc.page_content for doc in docs])
                    
                    # Generate answer
                    chain = prompt | llm | StrOutputParser()
                    answer = chain.invoke({"context": context, "question": question})
                    
                    st.write(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})

elif uploaded_file and not api_key:
    st.warning("Please enter your OpenAI API key in the sidebar.")
elif api_key and not uploaded_file:
    st.info("Please upload a PDF to get started.")
else:
    st.info("Enter your API key and upload a PDF to start chatting.")