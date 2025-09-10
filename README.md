# PDF-Q-A-Chatbot-with-RAG-FAISS
LangChain, FAISS, Sentence Transformers, Streamlit

A powerful AI-powered chatbot that can answer questions from uploaded PDF documents using Retrieval-Augmented Generation (RAG) architecture.
🚀 Features

PDF Document Processing: Upload multiple PDF files and extract text content
Semantic Search: Uses Sentence Transformers to create embeddings and FAISS for efficient similarity search
Conversational AI: Maintains conversation history and context across questions
Source Attribution: Shows relevant document excerpts that support the answers
User-Friendly Interface: Clean, modern Streamlit interface with real-time chat
Flexible Configuration: Adjustable chunk sizes, overlap, and embedding models

📋 Prerequisites

Python 3.8 or higher
4GB RAM minimum (8GB recommended for larger documents)
Internet connection (for downloading models on first run)

🛠️ Installation
1. Clone or Download the Project
bash# Create a new directory for your project
mkdir pdf-qa-chatbot
cd pdf-qa-chatbot

# Save the provided files in this directory
2. Set Up Virtual Environment (Recommended)
bash# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
3. Install Dependencies
bash
pip install -r requirements.txt

4. Configure API Keys

Get a free Hugging Face API token:

Go to https://huggingface.co/settings/tokens
Create an account if needed
Generate a new token

Create a .env file in your project directory and add your token:
bashHUGGINGFACEHUB_API_TOKEN=your_token_here

5. Run the Application
bash
streamlit run app.py
The application will open in your default web browser at http://localhost:8501
