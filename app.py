from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
import logging
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()

# Initialize the Flask app and configure CORS
app = Flask(__name__)
CORS(app, resources={"/api/*": {"origins": "*"}})

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
    site1 = WebBaseLoader("https://middleeastretailforum.com/")
    site2 = WebBaseLoader("https://middleeastretailforum.com/speakers-2024/")
    site3= WebBaseLoader("https://middleeastretailforum.com/partners-2024/")
    site4= WebBaseLoader("https://middleeastretailforum.com/awards/")
    site5= WebBaseLoader("https://middleeastretailforum.com/nomination-process/")
    site6= WebBaseLoader("https://middleeastretailforum.com/jury-2024/")
    site7= WebBaseLoader("https://middleeastretailforum.com/about-images-group/")
    site8= WebBaseLoader("https://middleeastretailforum.com/agenda-2024/")
    site9= WebBaseLoader("https://middleeastretailforum.com/award-categories/")
    site10= WebBaseLoader("https://middleeastretailforum.com/speakers-2023/")    
    document1 = loader.load()
    document2 = site1.load() + site2.load() + site3.load() + site4.load() + site5.load() + site6.load() + site7.load() + site8.load() + site9.load() + site10.load()
    document = document1 + document2
    text_splitter = RecursiveCharacterTextSplitter()  # Adjusted chunk size
    document_chunks = text_splitter.split_documents(document)
    return document_chunks

def get_context_retriever_chain():
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
        ("system", "You are a professional and friendly AI editor at Images RetailMe, your name is Noura, and your job is to provide assistance to the users who want to know about the Middle East Retail Forum taking place on 26th September 2024. You help users find information with a precise and accurate attitude. You answer their queries in complete sentences and in a precise manner, not in points and not using any bold letters, within 100 tokens, to the point and very precise and short, based on the context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    retriever_chain = get_context_retriever_chain()
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    response = conversation_rag_chain.invoke({
        "chat_history": [],
        "input": user_input
    })
    return response['answer']

@app.route('/api/upload_doc', methods=['POST'])
def upload_pdf():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        if file:
            file_path = f"temp_{file.filename}"
            file.save(file_path)
            executor.submit(async_load_and_save, file_path)
            return jsonify({"message": "PDF is being processed. Please check back later."})
    except Exception as e:
        logging.error(f"Error during PDF upload: {e}")
        return jsonify({"error": str(e)}), 500

def async_load_and_save(file_path, websites):
    try:
        document_chunks = load_document_chunks(file_path, websites)
        save_vector_store(document_chunks)
        os.remove(file_path)
        logging.info("PDF processed successfully and vector store updated.")
    except Exception as e:
        logging.error(f"Error during document processing: {e}")

@app.route('/api/ask', methods=['POST'])
def chat():
    global vector_store
    data = request.json
    user_input = data.get('question')

    # Ensure vector store is initialized
    if vector_store is None:
        initialize_vector_store()

    if vector_store is None:
        return jsonify({"error": "The vector store is not ready yet. Please upload a document first."}), 503

    response = get_response(user_input)
    return jsonify({"response": response})

if __name__ == '__main__':
    initialize_vector_store()
    app.run(debug=False)
