from flask import Flask, request, jsonify, abort ,  render_template , session
from flask_cors import CORS
import pyodbc
from dotenv import load_dotenv
import os
import logging
import bcrypt
import uuid
import random
import requests
import jwt
from openai import OpenAI
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor
from langchain_community.document_loaders import WebBaseLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from flask_session import Session


# Load environment variables
load_dotenv()
client = OpenAI(api_key= os.getenv("OPENAI_API_KEY"))
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

app.config['SESSION_TYPE'] = 'filesystem'
app.config['SECRET_KEY'] = 'A6089145627GGHT'  # Replace with a secure key
Session(app)
EMAIL_API_URL = "https://api.waysdatalabs.com/api/EmailSender/SendMail"
EMAIL_API_KEY = "6A7339A3-E70B-4A8D-AA23-0264125F4959"

# Initialize ThreadPoolExecutor for asynchronous processing
executor = ThreadPoolExecutor(max_workers=4)

# Global variable for the vector store
vector_store = None


# Helper Functions
def send_email(recipient, subject, body):
    """Send email using the provided email API."""
    try:
        payload = {
            'Recipient': recipient,
            'Subject': subject,
            'Body': body,
            'ApiKey': EMAIL_API_KEY
        }
        response = requests.post(EMAIL_API_URL, data=payload)
        response.raise_for_status()
        logging.info(f"Email sent to {recipient}")
    except Exception as e:
        logging.error(f"Failed to send email to {recipient}: {e}")


def generate_jwt(user_id):
    """Generate a JWT token."""
    payload = {
        'user_id': user_id,
        'exp': datetime.utcnow() + timedelta(hours=10)  # Token expires in 1 hour
    }
    return jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')


def validate_jwt(token):
    """Validate a JWT token."""
    try:
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        return payload['user_id']
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def authenticate():
    """JWT Middleware for Protected Routes."""
    token = request.headers.get('Authorization')
    if not token:
        return jsonify({"error": "Unauthorized. Please provide a valid token."}), 401

    user_id = validate_jwt(token)
    if not user_id:
        return jsonify({"error": "Unauthorized. Token is invalid or expired."}), 401

    return user_id


# APIs
@app.route('/api/verify-token', methods=['POST'])
def verify_token():
    """
    Verify the validity of a JWT token.
    """
    token = request.headers.get('Authorization')
    if not token:
        return jsonify({"error": "Missing JWT token in Authorization header."}), 401

    try:
        # Decode and validate the token
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        user_id = payload.get('user_id')
        exp_time = datetime.utcfromtimestamp(payload['exp']).strftime('%Y-%m-%d %H:%M:%S')
        return jsonify({
            "message": "Token is valid.",
            "user_id": user_id,
            "expires_at": exp_time
        }), 200
    except jwt.ExpiredSignatureError:
        return jsonify({"error": "JWT token has expired. Please log in again."}), 401
    except jwt.InvalidTokenError:
        return jsonify({"error": "Invalid JWT token."}), 401


@app.route('/api/register', methods=['POST'])
def register_user():
    """Register a new user and generate an OTP for email validation."""
    data = request.json
    name = data.get('name')
    email = data.get('email')
    phone_number = data.get('phone_number')

    if not name or not email or not phone_number:
        return jsonify({"error": "Name, email, and phone number are required."}), 400

    try:
        connection = pyodbc.connect(DB_CONNECTION_STRING)
        cursor = connection.cursor()

        # Check if email already exists
        cursor.execute("SELECT COUNT(*) FROM tb_MS_User WHERE email = ?", (email,))
        user_exists = cursor.fetchone()[0]
        if user_exists:
            return jsonify({"error": "User with this email already exists."}), 400

        # Generate OTP
        otp = str(random.randint(100000, 999999))
        otp_expiry = datetime.now() + timedelta(minutes=5)

        # Insert OTP into tb_UserOTP
        cursor.execute("INSERT INTO tb_UserOTP (email, otp, expiry_at) VALUES (?, ?, ?)", (email, otp, otp_expiry))
        connection.commit()

        # Send OTP Email
        otp_email_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Your OTP Code</title>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .otp {{ color: #2c7ae8; font-weight: bold; }}
            </style>
        </head>
        <body>
            <p>Dear {name},</p>
            <p>Your OTP code is <span class="otp">{otp}</span>. It is valid for 5 minutes.</p>
            <p>Best regards,<br>Team WaysAhead</p>
        </body>
        </html>
        """
        send_email(email, "Your OTP Code", otp_email_body)

        return jsonify({"message": "OTP sent successfully. Check your email for validation."}), 200
    except Exception as e:
        logging.error(f"Error during registration: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/validate-otp', methods=['POST'])
def validate_otp():
    """
    Validate the OTP and create the user.
    """
    data = request.json
    email = data.get('email')
    otp = data.get('otp')
    name = data.get('name')
    phone_number = data.get('phone_number')

    if not email or not otp or not name or not phone_number:
        return jsonify({"error": "Email, OTP, name, and phone number are required."}), 400

    try:
        connection = pyodbc.connect(DB_CONNECTION_STRING)
        cursor = connection.cursor()

        # Validate OTP
        cursor.execute("""
            SELECT otp_id, expiry_at, is_used FROM tb_UserOTP
            WHERE email = ? AND otp = ? AND is_deleted = 0
        """, (email, otp))
        otp_data = cursor.fetchone()

        if not otp_data:
            return jsonify({"error": "Invalid OTP."}), 400
        if otp_data[2]:  # is_used = True
            return jsonify({"error": "OTP already used."}), 400
        if otp_data[1] < datetime.now():  # expiry_at < current time
            return jsonify({"error": "OTP expired."}), 400

        # Mark OTP as used
        otp_id = otp_data[0]
        cursor.execute("""
            UPDATE tb_UserOTP 
            SET is_used = 1, updated_at = GETDATE()
            WHERE otp_id = ?
        """, (otp_id,))

        # Create User
        license_key = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=6))
        license_expiry_date = datetime.now() + timedelta(days=7)
        cursor.execute("""
            INSERT INTO tb_MS_User (name, email, phone_number, temporary_license_key, license_expiry_date)
            OUTPUT INSERTED.user_id
            VALUES (?, ?, ?, ?, ?)
        """, (name, email, phone_number, license_key, license_expiry_date))
        user_id = cursor.fetchone()[0]

        if not user_id:
            return jsonify({"error": "Failed to create user and retrieve user_id."}), 500

        # Update user_id in tb_UserOTP
        cursor.execute("""
            UPDATE tb_UserOTP 
            SET user_id = ?, updated_at = GETDATE()
            WHERE email = ? AND otp = ? AND is_used = 1 AND is_deleted = 0
        """, (user_id, email, otp))
        connection.commit()

        # Send Welcome Email
        welcome_email_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Welcome to WaysAhead</title>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .license-key {{ color: #2c7ae8; font-weight: bold; }}
            </style>
        </head>
        <body>
            <p>Dear {name},</p>
            <p>Welcome to GeoPlatform! Your temporary license key is <span class="license-key">{license_key}</span>.</p>
            <p>This key is valid for 7 days. Enjoy exploring the platform!</p>
            <p>Best regards,<br>Team WaysAhead</p>
        </body>
        </html>
        """
        send_email(email, "Welcome to GeoPlatform", welcome_email_body)

        return jsonify({
            "message": "Registration Successful",
            "license_key": license_key,
            "user_id": user_id
        }), 200
    except Exception as e:
        logging.error(f"Error during OTP validation: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/set-password', methods=['POST'])
def set_password():
    """Set a password for the user."""
    data = request.json
    user_id = data.get('user_id')
    password = data.get('password')

    if not user_id or not password:
        return jsonify({"error": "Email and password are required."}), 400

    try:
        connection = pyodbc.connect(DB_CONNECTION_STRING)
        cursor = connection.cursor()

        # Check if the user exists
        cursor.execute("SELECT email FROM tb_MS_User WHERE user_id = ?", (user_id,))
        email = cursor.fetchone()
        if not email:
            return jsonify({"error": "User not found. Please register first."}), 404

        # Hash the password
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        # Update the password
        cursor.execute("UPDATE tb_MS_User SET password_hash = ? WHERE user_id = ?", (password_hash, user_id))
        connection.commit()

        return jsonify({"message": "Password set successfully."}), 200
    except Exception as e:
        logging.error(f"Error during password setting: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/login', methods=['POST'])
def login():
    """Authenticate user and generate JWT."""
    data = request.json
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400
    try:
        connection = pyodbc.connect(DB_CONNECTION_STRING)
        cursor = connection.cursor()
        # Fetch user details
        cursor.execute("""
            SELECT user_id, password_hash, temporary_license_key, license_expiry_date
            FROM tb_MS_User 
            WHERE email = ?
        """, (email,))
        user = cursor.fetchone()

        if not user:
            return jsonify({"error": "Invalid email. Please register first."}), 401

        user_id, password_hash, temporary_license_key, license_expiry_date = user

        # Check if password is not set
        if not password_hash:
            return jsonify({"error": "Password not set. Please set your password to log in."}), 403

        # Verify the provided password
        if not bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8')):
            return jsonify({"error": "Invalid password."}), 401

        # Check if the license has expired
        if license_expiry_date < datetime.now():
            return jsonify({"error": "License expired. Please contact support to renew your subscription."}), 403
        # Generate JWT token
        token = generate_jwt(user_id)
        return jsonify({
            "message": "Login successful.",
            "token": token,
            "license_key": temporary_license_key,
            "user_id": user_id
        }), 200
    except Exception as e:
        logging.error(f"Error during login: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/preferences', methods=['POST'])
def save_preferences():
    """
    Save user preferences, generate insights using OpenAI, and send a personalized email.
    """
    data = request.json
    user_id = data['user_id']
    preferences = {
        "business_type": data['business_type'],
        "weekly_footfall": data['estimated_weekly_footfall'],
        "monthly_revenue": data['estimated_monthly_revenue'],
        "transaction_value": data['avg_transaction_value'],
        "audience_age": data['target_audience_age'],
        "audience_category": data['target_audience_category']
    }

    if not user_id or not preferences:
        return jsonify({"error": "User ID and preferences are required."}), 400

    try:
        connection = pyodbc.connect(DB_CONNECTION_STRING)
        cursor = connection.cursor()

        # Insert preferences into tb_UserPreferences
        cursor.execute("""
            INSERT INTO tb_UserPreferences (user_id, business_type, estimated_weekly_footfall, 
                estimated_monthly_revenue, avg_transaction_value, target_audience_age, target_audience_category)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (user_id, preferences["business_type"], preferences["weekly_footfall"],
              preferences["monthly_revenue"], preferences["transaction_value"], 
              preferences["audience_age"], preferences["audience_category"]))

        connection.commit()

        # Prepare the input prompt for OpenAI model
        comparison_prompt = f"""
        Based on the following preferences:
        - Business Type: {preferences["business_type"]}
        - Estimated Weekly Footfall: {preferences["weekly_footfall"]}
        - Estimated Monthly Revenue: {preferences["monthly_revenue"]}
        - Average Transaction Value: {preferences["transaction_value"]}
        - Target Audience Age: {preferences["audience_age"]}
        - Target Audience Category: {preferences["audience_category"]}

        Generate detailed insights and suggestions to optimize their business strategy.
        """
        insights_response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            response_format={"type": "text"},
            temperature=0.7,
            max_tokens=250,
            messages=[
                {"role": "system", "content": "You are a helpful assistant generating actionable business insights."},
                {"role": "user", "content": comparison_prompt}
            ]
        )

        # Extract insights from the OpenAI response
        insights = insights_response['choices'][0]['message']['content'].strip()

        # Fetch user email
        cursor.execute("SELECT email FROM tb_MS_User WHERE user_id = ?", (user_id,))
        email = cursor.fetchone()[0]

        # Generate and send personalized email
        email_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Personalized Insights</title>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .highlight {{ color: #2c7ae8; font-weight: bold; }}
            </style>
        </head>
        <body>
            <p>Dear User,</p>
            <p>Thank you for sharing your preferences. Based on your inputs, here are some insights from our AI engine:</p>
            <blockquote style="font-style: italic; color: #555;">{insights}</blockquote>
            <p>If you have further questions, feel free to contact our sales team at <a href="mailto:sales@geoplatform.com">sales@geoplatform.com</a>.</p>
            <p>Best regards,<br>Team GeoPlatform</p>
        </body>
        </html>
        """
        send_email(email, "Your Personalized Insights from GeoPlatform", email_body)
        return jsonify({"message": "Preferences saved successfully. Insights emailed."}), 200
    except Exception as e:
        logging.error(f"Error during preferences saving: {e}")
        return jsonify({"error": str(e)}), 500

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
