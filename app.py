from flask import Flask, request, jsonify, abort ,  render_template
from flask_cors import CORS
import pyodbc
from dotenv import load_dotenv
import os
import logging
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor
from langchain_community.document_loaders import WebBaseLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


# Load environment variables
load_dotenv()

# Initialize the Flask app and configure CORS
app = Flask(__name__)
CORS(app, resources={"/api/*": {"origins": "*"}})

# UAE timezone setup
UAE_TZ = timezone(timedelta(hours=4))

# Database connection string
DB_CONNECTION_STRING = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=103.239.89.99,21433;DATABASE=RetailMEApp_DB;UID=RetailMEAppUsr;PWD=App*Retail8usr"

# Directory to persist the vector store
VECTOR_STORE_DIR = "persistent_vector_store"

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize ThreadPoolExecutor for asynchronous processing
executor = ThreadPoolExecutor(max_workers=4)

# Global variable for the vector store
vector_store = None

def initialize_vector_store():
    """Initialize the vector store from the persistent directory, if it exists."""
    global vector_store
    if os.path.exists(VECTOR_STORE_DIR):
        vector_store = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=OpenAIEmbeddings())
    else:
        vector_store = None

def save_vector_store(document_chunks):
    """Save the vector store after creating it from document chunks."""
    global vector_store
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings(), persist_directory=VECTOR_STORE_DIR)
    vector_store.persist()

def load_document_chunks(file_path):
    """Load and split documents to handle large files and multiple websites."""
    loader = PyMuPDFLoader(file_path)
    site1 = WebBaseLoader("https://www.visitdubai.com/en/business-in-dubai/industries/retail")
    site2 = WebBaseLoader("https://www.khaleejtimes.com/business/retail")
    site3 = WebBaseLoader("https://gulfnews.com/business/retail")
    site4 = WebBaseLoader("https://www.retail-insight-network.com/")
    site5 = WebBaseLoader("https://saudiretailforum.com/")

    document1 = loader.load()
    document2 = site1.load() + site2.load() + site3.load() + site4.load() + site5.load() 
    document = document1 + document2

    text_splitter = RecursiveCharacterTextSplitter()  # Adjusted chunk size
    document_chunks = text_splitter.split_documents(document)
    return document_chunks

def get_context_retriever_chain(session_id):
    llm = ChatOpenAI()
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Based on the conversation, generate a search query to find relevant information.")
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0.7)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are GeoBot, an AI assistant for geospatial analytics and insights in the Middle East and Dubai. Your role is to guide users through geospatial trends, retail data, and location insights, offering clear and accurate information. Respond within 100 tokens, always maintaining a professional tone. Provide precise and actionable insights based on the context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input, session_id, start_time):
    retriever_chain = get_context_retriever_chain(session_id)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    chat_history = load_chat_history(session_id)
    formatted_chat_history = [
        {"role": "user", "content": entry["user_input"]} if i % 2 == 0 else {"role": "assistant", "content": entry["bot_response"]}
        for i, entry in enumerate(chat_history)
    ]

    response = conversation_rag_chain.invoke({
        "chat_history": formatted_chat_history,
        "input": user_input
    })

    save_chat_history(session_id, user_input, response['answer'], start_time)
    return response['answer']


def log_api_call(endpoint, status_code, response_time):
    """Log API call details to the database."""
    connection = pyodbc.connect(DB_CONNECTION_STRING)
    cursor = connection.cursor()
    cursor.execute(
        "INSERT INTO tb_Geobot_APILog (api_endpoint, status_code, response_time, timestamp) VALUES (?, ?, ?, ?)",
        (endpoint, status_code, response_time, datetime.now(UAE_TZ))
    )
    connection.commit()
    connection.close()

def authenticate_api_key(api_key):
    """Check if the provided API key is valid and active."""
    connection = pyodbc.connect(DB_CONNECTION_STRING)
    cursor = connection.cursor()
    cursor.execute("SELECT is_active FROM tb_Geobot_APIkey WHERE api_key = ?", (api_key,))
    result = cursor.fetchone()
    connection.close()
    return result and result[0]

def save_chat_history(session_id, user_input, bot_response, start_time):
    """Save user input and bot response to the database."""
    connection = pyodbc.connect(DB_CONNECTION_STRING)
    cursor = connection.cursor()
    response_time = (datetime.now(UAE_TZ) - start_time).total_seconds()
    cursor.execute(
        "INSERT INTO tb_Geobot (session_id, user_input, bot_response, response_time, timestamp) VALUES (?, ?, ?, ?, ?)",
        (session_id, user_input, bot_response, response_time, datetime.now(UAE_TZ))
    )
    connection.commit()
    connection.close()

def load_chat_history(session_id):
    """Load chat history for a given session ID."""
    connection = pyodbc.connect(DB_CONNECTION_STRING)
    cursor = connection.cursor()
    cursor.execute("SELECT user_input, bot_response FROM tb_Geobot WHERE session_id = ? ORDER BY timestamp ASC", (session_id,))
    chat_history = cursor.fetchall()
    connection.close()

    # Convert each row to a dictionary
    chat_history_list = [{"user_input": row.user_input, "bot_response": row.bot_response} for row in chat_history]
    return chat_history_list

@app.route('/api/upload_doc', methods=['POST'])
def upload_pdf():
    start_time = datetime.now(UAE_TZ)
    try:
        if 'file' not in request.files:
            log_api_call('/api/upload_doc', 400, (datetime.now(UAE_TZ) - start_time).total_seconds())
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            log_api_call('/api/upload_doc', 400, (datetime.now(UAE_TZ) - start_time).total_seconds())
            return jsonify({"error": "No selected file"}), 400

        if file:
            file_path = f"temp_{file.filename}"
            file.save(file_path)
            executor.submit(async_load_and_save, file_path)
            log_api_call('/api/upload_doc', 200, (datetime.now(UAE_TZ) - start_time).total_seconds())
            return jsonify({"message": "Document is being processed. Please check back later."})

    except Exception as e:
        log_api_call('/api/upload_doc', 500, (datetime.now(UAE_TZ) - start_time).total_seconds())
        logging.error(f"Error during document upload: {e}")
        return jsonify({"error": str(e)}), 500

def async_load_and_save(file_path):
    try:
        document_chunks = load_document_chunks(file_path)
        save_vector_store(document_chunks)
        os.remove(file_path)
        logging.info("Document processed successfully and vector store updated.")
    except Exception as e:
        logging.error(f"Error during document processing: {e}")

@app.route('/api/ask', methods=['POST'])
def chat():
    global vector_store
    start_time = datetime.now(UAE_TZ)
    try:
        api_key = request.headers.get('Authorization')
        if not authenticate_api_key(api_key):
            log_api_call('/api/ask', 401, (datetime.now(UAE_TZ) - start_time).total_seconds())
            return jsonify({"error": "Invalid API key"}), 401

        data = request.get_json()
        user_input = data.get('user_input')
        session_id = data.get('session_id')
        if vector_store is None:
            initialize_vector_store()
        if vector_store is None:
            return jsonify({"error": "The vector store is not ready yet. Please upload a document first."}), 503
        if not user_input or not session_id:
            log_api_call('/api/ask', 400, (datetime.now(UAE_TZ) - start_time).total_seconds())
            return jsonify({"error": "Missing input or session ID"}), 400

        response = get_response(user_input, session_id, start_time)
        log_api_call('/api/ask', 200, (datetime.now(UAE_TZ) - start_time).total_seconds())
        return jsonify({"response": response})

    except Exception as e:
        log_api_call('/api/ask', 500, (datetime.now(UAE_TZ) - start_time).total_seconds())
        logging.error(f"Error during chat processing: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    initialize_vector_store()
    app.run(debug=True)
