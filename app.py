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

# Global storage for vector store and initialization flag
vector_store = None
vector_store_ready = False

def load_document_chunks(file_path, websites):
    """Load and split documents to handle large files and multiple websites."""
    try:
        loader = PyMuPDFLoader(file_path)
        document1 = loader.load()

        document2 = []
        for site in websites:
            site_loader = WebBaseLoader(site)
            doc_chunks = site_loader.load()
            document2.extend(doc_chunks)

        combined_documents = document1 + document2
        text_splitter = RecursiveCharacterTextSplitter()
        document_chunks = text_splitter.split_documents(combined_documents)
        return document_chunks
    except Exception as e:
        raise

def get_vectorstore_from_chunks(document_chunks):
    """Generate a vector store from the document chunks."""
    try:
        vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())
        return vector_store
    except Exception as e:
        raise

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
    llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0.7)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional and friendly AI editor at Images RetailMe, your name is Noura, and your job is to provide assistance to the users who want to know about the Middle East Retail Forum taking place on 26th September 2024. You help users find information with a precise and accurate attitude. You answer their queries in complete sentences and in a precise manner, not in points and not using any bold letters, within 100 tokens, to the point and very precise and short, based on the context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input, vector_store):
    if vector_store is None:
        return "The vector store has not been initialized. Please upload a document first."

    retriever_chain = get_context_retriever_chain(vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    response = conversation_rag_chain.invoke({
        "chat_history": [],
        "input": user_input
    })
    return response['answer']

@app.route('/api/upload_doc', methods=['POST'])
def upload_pdf():
    global vector_store, vector_store_ready
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
                "https://middleeastretailforum.com/partners-2023/",
                "https://middleeastretailforum.com/speakers-2023/",
                "https://middleeastretailforum.com/agenda-2023/",
                "https://middleeastretailforum.com/mrf-2023-post-show-report/",
                "https://middleeastretailforum.com/speakers-2022/",
                "https://middleeastretailforum.com/partners-2022/",
                "https://middleeastretailforum.com/agenda-2022/",
                "https://middleeastretailforum.com/companies-over-the-years/"
            ]
            document_chunks = load_document_chunks(file_path, websites)
            vector_store = get_vectorstore_from_chunks(document_chunks)
            vector_store_ready = True  # Mark vector store as ready
            os.remove(file_path)
            return jsonify({"message": "PDF processed successfully."})
    except Exception as e:
        vector_store_ready = False  # Ensure it's not marked ready if there's an error
        return jsonify({"error": str(e)}), 500

@app.route('/api/ask', methods=['POST'])
def chat():
    global vector_store, vector_store_ready
    data = request.json
    user_input = data.get('question')

    if not vector_store_ready:
        return jsonify({"error": "The vector store is not ready yet. Please try again after the document is uploaded and processed."}), 503

    response = get_response(user_input, vector_store)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=False)
