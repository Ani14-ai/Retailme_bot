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
def get_vectorstore_from_pdf(file_path, web_urls):
    pdf_loader = PyMuPDFLoader(file_path)
    pdf_document = pdf_loader.load()
    web_documents = []
    for url in web_urls:
        web_loader = WebBaseLoader(url)
        web_documents.extend(web_loader.load())
    all_documents = pdf_document + web_documents
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(all_documents)
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())  
    return vector_store
web_urls = [
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
    llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18" , temperature = 0.7)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional and friendly AI editor at Images RetailMe , your name is Noura and Your job is to provide assistance to the users who want to know about the Middle East Retail Forum taking place on 26th September 2024 . You help users find information with a precise and accurate attitude. You answer their queries in complete sentences and in precise manner , not in points and not using any bold letters, within 100 tokens,to the point and very precise and short, based on the context:\n\n{context}"),
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
chat_history = [AIMessage(content="Hello! I'm a friendly and professional  AI editor at Images RetailME, my name is Noura. How can I assist you today?")]


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
            vector_store = get_vectorstore_from_pdf(file_path , web_urls)
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
