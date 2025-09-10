import streamlit as st
import os
import tempfile
from pathlib import Path
import pickle
from datetime import datetime

# PDF processing
from PyPDF2 import PdfReader

# LangChain components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFaceHub
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import Document

# Initialize session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()

# Page configuration
st.set_page_config(
    page_title="PDF Q&A Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .stButton > button {
        background-color: #667eea;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #764ba2;
        transform: translateY(-2px);
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    </style>
""", unsafe_allow_html=True)

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page_num, page in enumerate(pdf_reader.pages):
        page_text = page.extract_text()
        if page_text:
            text += f"\n--- Page {page_num + 1} ---\n{page_text}"
    return text

def create_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    """Split text into manageable chunks with overlap"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks

def create_vectorstore(text_chunks, embeddings):
    """Create FAISS vector store from text chunks"""
    documents = [Document(page_content=chunk) for chunk in text_chunks]
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    """Create conversational retrieval chain"""
    # Using a free model from HuggingFace Hub
    # You can replace this with OpenAI or other LLMs
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={"temperature": 0.7, "max_length": 512}
    )
    
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True
    )
    
    return conversation_chain

def process_user_question(question):
    """Process user question and display response"""
    if st.session_state.conversation is None:
        st.warning("Please upload and process a PDF document first!")
        return
    
    # Get response from the chain
    with st.spinner("Thinking..."):
        response = st.session_state.conversation({
            'question': question,
            'chat_history': st.session_state.chat_history
        })
    
    # Update chat history
    st.session_state.chat_history.append((question, response['answer']))
    
    # Display the conversation
    for i, (q, a) in enumerate(st.session_state.chat_history):
        with st.container():
            st.markdown(f'<div class="chat-message user-message">👤 **You:** {q}</div>', 
                       unsafe_allow_html=True)
            st.markdown(f'<div class="chat-message bot-message">🤖 **Assistant:** {a}</div>', 
                       unsafe_allow_html=True)
            
            # Show source documents for the latest response
            if i == len(st.session_state.chat_history) - 1 and 'source_documents' in response:
                with st.expander("📄 View Source Excerpts"):
                    for j, doc in enumerate(response['source_documents'][:3]):
                        st.markdown(f"**Source {j+1}:**")
                        st.markdown(f"```{doc.page_content[:500]}...```")

# Main UI
st.markdown('<h1 class="main-header">📚 PDF Q&A Chatbot with RAG</h1>', unsafe_allow_html=True)
st.markdown("### Powered by LangChain, FAISS & Sentence Transformers")

# Sidebar for PDF upload and settings
with st.sidebar:
    st.header("📁 Document Upload")
    
    # PDF upload
    pdf_files = st.file_uploader(
        "Upload PDF Documents",
        type=['pdf'],
        accept_multiple_files=True,
        help="Select one or more PDF files to analyze"
    )
    
    # Advanced settings
    with st.expander("⚙️ Advanced Settings"):
        chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100)
        chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200, 50)
        model_name = st.selectbox(
            "Embedding Model",
            ["sentence-transformers/all-MiniLM-L6-v2",
             "sentence-transformers/all-mpnet-base-v2",
             "sentence-transformers/paraphrase-MiniLM-L6-v2"]
        )
    
    # Process button
    if st.button("🚀 Process Documents", type="primary"):
        if pdf_files:
            with st.spinner("Processing PDFs..."):
                # Initialize embeddings
                embeddings = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={'device': 'cpu'}
                )
                
                all_chunks = []
                
                # Process each PDF
                progress_bar = st.progress(0)
                for idx, pdf_file in enumerate(pdf_files):
                    if pdf_file.name not in st.session_state.processed_files:
                        # Extract text
                        text = extract_text_from_pdf(pdf_file)
                        
                        # Create chunks
                        chunks = create_text_chunks(text, chunk_size, chunk_overlap)
                        all_chunks.extend(chunks)
                        
                        # Add to processed files
                        st.session_state.processed_files.add(pdf_file.name)
                    
                    progress_bar.progress((idx + 1) / len(pdf_files))
                
                # Create or update vectorstore
                if all_chunks:
                    if st.session_state.vectorstore is None:
                        st.session_state.vectorstore = create_vectorstore(all_chunks, embeddings)
                    else:
                        # Add new documents to existing vectorstore
                        new_docs = [Document(page_content=chunk) for chunk in all_chunks]
                        st.session_state.vectorstore.add_documents(new_docs)
                    
                    # Create conversation chain
                    st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore)
                    
                    st.success(f"✅ Successfully processed {len(pdf_files)} document(s)!")
                    st.info(f"📊 Total chunks in knowledge base: {len(all_chunks)}")
        else:
            st.error("Please upload at least one PDF file!")
    
    # Clear conversation button
    if st.button("🗑️ Clear Conversation"):
        st.session_state.chat_history = []
        st.rerun()
    
    # Statistics
    if st.session_state.vectorstore:
        st.markdown("---")
        st.markdown("### 📊 Statistics")
        st.metric("Processed Files", len(st.session_state.processed_files))
        st.metric("Total Chunks", st.session_state.vectorstore.index.ntotal)

# Main chat interface
col1, col2 = st.columns([3, 1])

with col1:
    # Chat history display
    if st.session_state.chat_history:
        st.markdown("### 💬 Conversation History")
        for q, a in st.session_state.chat_history:
            with st.container():
                st.markdown(f'<div class="chat-message user-message">👤 **You:** {q}</div>', 
                           unsafe_allow_html=True)
                st.markdown(f'<div class="chat-message bot-message">🤖 **Assistant:** {a}</div>', 
                           unsafe_allow_html=True)
    
    # Question input
    with st.form(key="question_form", clear_on_submit=True):
        user_question = st.text_input(
            "Ask a question about your documents:",
            placeholder="What is the main topic of the document?",
            key="question_input"
        )
        submit_button = st.form_submit_button("Send 📤")
        
        if submit_button and user_question:
            process_user_question(user_question)
            st.rerun()

with col2:
    # Suggested questions
    st.markdown("### 💡 Suggested Questions")
    suggested_questions = [
        "What is the main topic?",
        "Summarize the key points",
        "What are the conclusions?",
        "List the main findings",
        "What methodology was used?"
    ]
    
    for question in suggested_questions:
        if st.button(question, key=f"suggested_{question}"):
            process_user_question(question)
            st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        Built with ❤️ using LangChain, FAISS, and Streamlit<br>
        <small>Note: For production use, consider using OpenAI API or other commercial LLMs for better responses</small>
    </div>
    """,
    unsafe_allow_html=True
)