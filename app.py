from flask import Flask, request, jsonify, render_template
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
load_dotenv()
app = Flask(__name__)
CORS(app, resources={"/api/*": {"origins": "*"}})
api_key=os.getenv("OPENAI_API_KEY")
os.environ["openai_api_key"] = api_key

def get_vectorstore_from_pdf(file_path):
    loader = PyMuPDFLoader(file_path)
    site1 = WebBaseLoader("https://www.imagesretailme.com/")
    site2 = WebBaseLoader("https://www.imagesretailme.com/category/leadership/")
    site3= WebBaseLoader("https://www.imagesretailme.com/bookstore/")
    document1 = loader.load()
    document2 = site1.load() + site2.load() + site3.load()
    document=document1 + document2
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())
    return vector_store

def get_context_retriever_chain(vector_store):
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
    llm = ChatOpenAI()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional assistant for RetailME. You help users find information with a precise and accurate attitude. You answer their queries in complete sentences within 100 tokens, based on the context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input, vector_store, chat_history):
    retriever_chain = get_context_retriever_chain(vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": chat_history,
        "input": user_input
    })
    return response['answer']


vector_store = None
chat_history = [AIMessage(content="Hello! I'm your friendly and professional assistant for RetailME. How can I assist you today?")]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/<path:path>')
def serve_page(path):
    return render_template(path)

@app.route('/api/upload_doc', methods=['POST'])
def upload_pdf():
    global vector_store
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        if file:
            file_path = f"temp_{file.filename}"
            file.save(file_path)
            vector_store = get_vectorstore_from_pdf(file_path)
            os.remove(file_path)
            return jsonify({"message": "PDF processed successfully."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/ask', methods=['POST'])
def chat():
    global chat_history, vector_store
    data = request.json
    user_input = data.get('question')
    response = get_response(user_input, vector_store, chat_history)    
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response))    
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=False)
