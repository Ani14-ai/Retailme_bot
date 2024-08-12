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

load_dotenv()

# Initialize the Flask app and configure CORS
app = Flask(__name__)
CORS(app, resources={"/api/*": {"origins": "*"}})

# Directory to persist the vector store
VECTOR_STORE_DIR = "persistent_vector_store"

# Initialize the vector store as None initially
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

def load_document_chunks(file_path, websites):
    """Load and split documents to handle large files and multiple websites."""
    loader = PyMuPDFLoader(file_path)
    document1 = loader.load()
    document2 = []
    for site in websites:
        site_loader = WebBaseLoader(site)
        doc_chunks = site_loader.load()
        document2.extend(doc_chunks)
        print(document2)

    combined_documents = document1 + document2
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(combined_documents)
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
            websites = [
                "https://middleeastretailforum.com/",
                "https://middleeastretailforum.com/download-brochure/",
                "https://middleeastretailforum.com/mrf-showreel/",
                "https://middleeastretailforum.com/speakers-over-the-years/",
                "https://middleeastretailforum.com/speakers-2024/",
                "https://middleeastretailforum.com/agenda-2024/speakers-2024/",
                "https://middleeastretailforum.com/partners-2024/",
                "https://middleeastretailforum.com/nomination-process/",
                "https://middleeastretailforum.com/award-categories/",
                "https://middleeastretailforum.com/jury-2024/",
                "https://middleeastretailforum.com/companies-over-the-years/"
            ]
            document_chunks = load_document_chunks(file_path, websites)
            save_vector_store(document_chunks)
            os.remove(file_path)
            return jsonify({"message": "PDF processed successfully."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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
