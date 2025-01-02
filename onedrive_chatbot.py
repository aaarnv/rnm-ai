import os
import requests
import sqlite3
import mimetypes
import io
from flask import Flask, request, redirect, session, render_template, jsonify
from msal import ConfidentialClientApplication
from PyPDF2 import PdfReader
import pandas as pd
import tiktoken
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings, OpenAI as LangchainOpenAI
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, UnstructuredExcelLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from square.client import Client
import json
from datetime import datetime, timedelta
import pytz

# App configuration 
CLIENT_ID = "8c3d5655-72bd-4c68-83a8-5f3ed8e5dd64"
CLIENT_SECRET = "Jmp8Q~s.Cm5nKx7gOVLnUL~Q9SgyF_c~PekQ8aaV"
AUTHORITY = "https://login.microsoftonline.com/common"
REDIRECT_URI = "http://localhost:5001/getAToken"
SCOPES = ["Files.ReadWrite.All", "User.Read"]
GRAPH_API_ENDPOINT = "https://graph.microsoft.com/v1.0"
OPENAI_API_KEY = "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" 
SQUARE_APPLICATION_ID = "sq0idp-vXoDPhpQnkrjkmQUK2xlBA"
SQUARE_APPLICATION_SECRET = "sq0csp-5EGutVtHN2ys2n06V29rjlLvM974fazrvTwiDXaO5sc"
SQUARE_REDIRECT_URI = "http://localhost:5001/squareauth"

LOCATION_IDS = {
    'st james': '0SKT76TMJS9GV',
    'pitt st': 'LJZ754B8QFEVZ',
    'darlinghurst': 'LFWCQDMP4Y92A',
    'redfern': 'L60V3VN61S9H9',
    # 'central kitchen': 'LCQBPXNQHCCQ8',
    'macquarie park': 'L62J4HVQ7X74M',
    'mq 2': 'L4TG236P6BFG8'
}

class IntegratedDocumentChatbot:
    def __init__(self, openai_api_key, db_path="onedrive_files.db"):
        """
        Initialize the chatbot with OpenAI API key and database connection
        """
        self.openai_api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key

        # Initialize OpenAI components
        self.embeddings = OpenAIEmbeddings()
        self.llm = LangchainOpenAI(temperature=0.3, max_tokens=100)
        self.client = OpenAI(api_key=openai_api_key)

        # Database connection and document management
        self.db_path = db_path
        self.loaded_documents = []
        self.manually_uploaded_documents = []
        
        # Vector store management
        self.vector_store = None
        self.last_document_count = 0

    def init_square_client(self, access_token):
        return Client(
            access_token=access_token,
            environment='production'  # Change to 'production' for live data
        )

    def load_pdf(self, pdf_file):
        """
        Load PDF file and split into chunks for manual upload
        """
        with open("temp.pdf", "wb") as f:
            f.write(pdf_file.read())

        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        
        # Add to manually uploaded documents
        self.manually_uploaded_documents.extend(texts)

        # Save to database for persistence
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        for i, doc in enumerate(texts):
            cursor.execute("""
                INSERT OR REPLACE INTO files (id, name, content, last_modified)
                VALUES (?, ?, ?, ?)
            """, (f"manual_pdf_{i}", "Uploaded PDF", doc.page_content, None))
        conn.commit()
        conn.close()

    def load_csv(self, csv_file):
        """
        Load CSV file for manual upload
        """
        with open("temp.csv", "wb") as f:
            f.write(csv_file.read())

        loader = CSVLoader(file_path="temp.csv")
        documents = loader.load()
        
        # Add to manually uploaded documents
        self.manually_uploaded_documents.extend(documents)

        # Save to database for persistence
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        for i, doc in enumerate(documents):
            cursor.execute("""
                INSERT OR REPLACE INTO files (id, name, content, last_modified)
                VALUES (?, ?, ?, ?)
            """, (f"manual_csv_{i}", "Uploaded CSV", doc.page_content, None))
        conn.commit()
        conn.close()

    def load_excel(self, excel_file):
        """
        Load Excel file for manual upload
        """
        with open("temp.xlsx", "wb") as f:
            f.write(excel_file.read())

        try:
            excel_file = pd.ExcelFile("temp.xlsx")
            all_sheets_content = []
            
            for sheet_name in excel_file.sheet_names:
                df = excel_file.parse(sheet_name)
                sheet_text = f"Sheet: {sheet_name}\n{df.to_string()}"
                
                text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                sheet_docs = text_splitter.create_documents([sheet_text])
                
                all_sheets_content.extend(sheet_docs)
            
            # Add to manually uploaded documents
            self.manually_uploaded_documents.extend(all_sheets_content)

            # Save to database for persistence
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            for i, doc in enumerate(all_sheets_content):
                cursor.execute("""
                    INSERT OR REPLACE INTO files (id, name, content, last_modified)
                    VALUES (?, ?, ?, ?)
                """, (f"manual_excel_{i}", f"Uploaded Excel - {doc.metadata.get('source', 'Unknown')}", doc.page_content, None))
            
            print(f"Loaded {len(all_sheets_content)} chunks from Excel file.")

            conn.commit()
            conn.close()

        except Exception as e:
            print(f"Error processing Excel file: {e}")

    def fetch_all_documents(self):
        """
        Combine OneDrive and manually uploaded documents
        """
        # Fetch database documents
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name, content FROM files")
        files = cursor.fetchall()
        conn.close()

        # Clear previous loaded documents
        self.loaded_documents = []

        # Add OneDrive files
        for name, content in files:
            doc = type('Document', (), {
                'page_content': f"File: {name}\nContent: {content}",
                'metadata': {'source': name}
            })
            self.loaded_documents.append(doc)
            print(f"Fetched document: {name}")

        # Add manually uploaded documents
        self.loaded_documents.extend(self.manually_uploaded_documents)

    def update_vector_store_if_needed(self):
        """
        Efficiently update vector store only when documents change
        """
        # Fetch current document count
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM files")
        current_doc_count = cursor.fetchone()[0]
        conn.close()

        # Check if documents have changed or first initialization
        if (self.vector_store is None or 
            current_doc_count != self.last_document_count or 
            len(self.manually_uploaded_documents) > 0):
            
            # Fetch all documents
            self.fetch_all_documents()
            
            # Recreate vector store
            if self.loaded_documents:
                # Add a unique ID to each document
                for i, doc in enumerate(self.loaded_documents):
                    doc.id = f"doc_{i}"
                
                self.vector_store = FAISS.from_documents(
                    self.loaded_documents, 
                    self.embeddings
                )
                
                # Update tracking
                self.last_document_count = current_doc_count
                print(f"Vector store updated with {len(self.loaded_documents)} documents.")

    def validate_token_limit(self, docs, question):
        """
        Validate if the token limit is exceeded.
        """
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        question_tokens = len(encoding.encode(question))
        doc_tokens = [len(encoding.encode(doc.page_content)) for doc in docs]

        print(f"Document tokens: {[len(encoding.encode(doc.page_content)) for doc in docs]}")
        total_tokens = question_tokens
        valid_docs = []

        for tokens, doc in zip(doc_tokens, docs):
            if total_tokens + tokens <= 4097 - 256:  # Reserve space for output
                total_tokens += tokens
                valid_docs.append(doc)
            else:
                print("Warning: File too large for processing.")
                break

        return valid_docs

    def answer_question(self, question, chat_history):
        """
        Unified method to handle both general and Square-related queries with proper chat history
        """
        # Simplified Square query detection
        is_square_query = 'square' in question.lower()
        
        try:
            if is_square_query:
                # Handle Square queries
                access_token = session.get('square_access_token')
                if not access_token:
                    return "Please connect to Square first to access sales data."

                client = self.init_square_client(access_token)
                return self.handle_square_query(question, client, chat_history)
            else:
                # Handle general queries using document search
                self.update_vector_store_if_needed()
                
                # Initialize context with relevant documents
                context = []
                if self.vector_store:
                    docs = self.vector_store.similarity_search(question, k=5)
                    docs = self.validate_token_limit(docs, question)
                    document_context = "\n\n".join([doc.page_content for doc in docs])
                    context.append({
                        "role": "system",
                        "content": f"Relevant information from documents:\n{document_context}"
                    })
                
                # Combine context, chat history, and current question
                full_conversation = context + chat_history + [{"role": "user", "content": question}]
                
                # Generate response
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=full_conversation
                )
                
                return response.choices[0].message.content
                
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            return f"An error occurred while processing your query: {str(e)}"
        
    def handle_square_query(self, question, client, chat_history=None):
        """
        Handle Square queries using chat history for context, with proper timezone handling.
        """
        # Extract date and location information using OpenAI with context from chat history
        context_messages = [
            {
                "role": "system", 
                "content": """You are a JSON generator that specializes in date and location parsing for Square data queries.
                Use context from previous messages when available. Return a valid JSON object with this structure:
                {
                    "start_date": "YYYY-MM-DD",
                    "end_date": "YYYY-MM-DD",
                    "location": "location_name",
                    "metric": "sales/items",
                    "item_name": null,
                    "relative_date": null,
                    "is_range": false
                }
                
                If a new location or date isn't specified, use the ones from the most recent previous query.
                If analyzing items, set metric to "items".
                """
            }
        ]
        
        # Add chat history for context
        if chat_history:
            context_messages.extend(chat_history)
        
        # Add current question
        context_messages.append({"role": "user", "content": question})
        
        date_response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=context_messages,
            temperature=0
        )
        
        try:
            query_params = json.loads(date_response.choices[0].message.content.strip())
            
            # Convert location name to ID
            location_id = None
            if query_params.get('location'):
                location_name = query_params['location'].lower()
                location_id = LOCATION_IDS.get(location_name)

            # Initialize Sydney timezone
            sydney_tz = pytz.timezone('Australia/Sydney')
            current_date = datetime.now(sydney_tz)
            
            # Handle date range parsing with timezone awareness
            if query_params.get('is_range'):
                start_time = sydney_tz.localize(datetime.strptime(query_params['start_date'], '%Y-%m-%d'))
                end_time = sydney_tz.localize(datetime.strptime(query_params['end_date'], '%Y-%m-%d'))
                end_time = end_time.replace(hour=23, minute=59, second=59)
            else:
                # Single date
                query_date = datetime.strptime(query_params['start_date'], '%Y-%m-%d')
                start_time = sydney_tz.localize(query_date.replace(hour=0, minute=0, second=0))
                end_time = sydney_tz.localize(query_date.replace(hour=23, minute=59, second=59))

            # Rest of the function remains the same...
            orders = []
            use_cache = self.should_use_cache(start_time, end_time)
            
            if use_cache:
                cached_orders = self.get_cached_orders(start_time, end_time, location_id)
                orders.extend(cached_orders)

            if not use_cache or datetime.now(sydney_tz).date() == end_time.date():
                real_time_orders = self.get_real_time_orders(client, start_time, end_time, location_id)
                orders.extend(real_time_orders)

            # Process orders and generate response...
            if orders:
                processed_data = self.process_square_orders(
                    orders, 
                    query_params.get('metric'),
                    query_params.get('item_name')
                )
                
                date_range_str = f"from {start_time.date()} to {end_time.date()}" if query_params.get('is_range') else f"on {start_time.date()}"
                context = f"Square data analysis for {query_params.get('location', 'all locations')} {date_range_str}:\n{json.dumps(processed_data, indent=2)}"
                
                response_messages = [
                    {"role": "system", "content": "You are a helpful assistant analyzing Square sales data. Provide specific insights and answer the question based on the data provided. When referring to previous context, be explicit about which location and date range you're discussing."},
                    {"role": "system", "content": context}
                ]
                
                if chat_history:
                    response_messages.extend(chat_history)
                
                response_messages.append({"role": "user", "content": question})
                
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=response_messages
                )
                
                return response.choices[0].message.content
            else:
                date_range_str = f"between {start_time.date()} and {end_time.date()}" if query_params.get('is_range') else f"on {start_time.date()}"
                return f"No orders found for {query_params.get('location', 'the specified location')} {date_range_str}"

        except json.JSONDecodeError as e:
            print("JSON Decode Error:", str(e))
            return "Error: Invalid response format from date parser"
        except Exception as e:
            print("General Error:", str(e))
            return f"Error processing Square query: {str(e)}"
            
        
    def should_use_cache(self, start_time, end_time):
        """
        Determine if we should use cached data based on the query date range
        """
        cache_threshold = datetime.now() - timedelta(days=7)

        temp_return = False
        return temp_return

    def get_cached_orders(self, start_time, end_time, location_id=None):
        """
        Retrieve orders from local cache within the specified date range
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = """
            SELECT content 
            FROM files 
            WHERE name LIKE 'Square Order%'
            AND json_extract(content, '$.created_at') BETWEEN ? AND ?
        """
        
        params = [start_time.isoformat(), end_time.isoformat()]

        if location_id:
            query += " AND json_extract(content, '$.location_id') = ?"
            params.append(location_id)

        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()

        return [json.loads(row[0]) for row in results]

    def get_real_time_orders(self, client, start_time, end_time, location_id=None):
        """
        Fetch real-time orders from Square API, handling pagination and Sydney timezone.
        
        Args:
            client: Square client instance
            start_time: datetime object in local time
            end_time: datetime object in local time
            location_id: Optional specific location ID to query
        """
        location_ids = [location_id] if location_id else list(LOCATION_IDS.values())
        orders = []
        cursor = None
        
        # Convert times to Sydney timezone
        sydney_tz = pytz.timezone('Australia/Sydney')
        
        # Ensure input times are aware of timezone
        if start_time.tzinfo is None:
            start_time = sydney_tz.localize(start_time)
        if end_time.tzinfo is None:
            end_time = sydney_tz.localize(end_time)
        
        # Format times in RFC 3339 format with Sydney offset
        start_time_str = start_time.isoformat()
        end_time_str = end_time.isoformat()

        while True:
            body = {
                'location_ids': location_ids,
                'query': {
                    'filter': {
                        'date_time_filter': {
                            'created_at': {
                                'start_at': start_time_str,
                                'end_at': end_time_str
                            }
                        }
                    }
                },
                'limit': 500
            }

            if cursor:
                body['cursor'] = cursor

            result = client.orders.search_orders(body=body)
            
            if result.is_success():
                batch_orders = result.body.get('orders', [])
                orders.extend(batch_orders)
                cursor = result.body.get('cursor')
                
                print(f"Retrieved {len(batch_orders)} orders. Total so far: {len(orders)}")

                # Stop if there are no more pages
                if not cursor:
                    break
            else:
                raise Exception(f"Square API Error: {result.errors}")

        return orders


    def process_square_orders(self, orders, metric_type='sales', item_filter=None):
        """
        Process Square orders based on query type and filters
        Returns data with location names instead of IDs
        """
        # Create reverse mapping of location IDs to names
        location_id_to_name = {id: name.title() for name, id in LOCATION_IDS.items()}
        
        results = {
            'total_sales': 0,
            'location_breakdown': {},
            'item_breakdown': {},
            'order_count': len(orders)
        }
        
        for order in orders:
            location_id = order.get('location_id')
            location_name = location_id_to_name.get(location_id, 'Unknown Location')
            total_money = float(order['total_money']['amount']) / 100
            
            # Update location totals using location name instead of ID
            if location_name not in results['location_breakdown']:
                results['location_breakdown'][location_name] = {
                    'total_sales': 0,
                    'order_count': 0,
                    'items': {}
                }
                
            results['total_sales'] += total_money
            results['location_breakdown'][location_name]['total_sales'] += total_money
            results['location_breakdown'][location_name]['order_count'] += 1
            
            # Process line items
            for item in order.get('line_items', []):
                item_name = item.get('name', 'Unknown Item')
                quantity = int(item.get('quantity', 1))
                item_total = float(item.get('total_money', {}).get('amount', 0)) / 100
                
                # Skip if item filter is set and doesn't match
                if item_filter and item_filter.lower() not in item_name.lower():
                    continue
                    
                if item_name not in results['item_breakdown']:
                    results['item_breakdown'][item_name] = {
                        'total_quantity': 0,
                        'total_sales': 0,
                        'location_breakdown': {}
                    }
                    
                results['item_breakdown'][item_name]['total_quantity'] += quantity
                results['item_breakdown'][item_name]['total_sales'] += item_total
                
                # Update location-specific item counts using location name
                if location_name not in results['item_breakdown'][item_name]['location_breakdown']:
                    results['item_breakdown'][item_name]['location_breakdown'][location_name] = {
                        'quantity': 0,
                        'sales': 0
                    }
                
                results['item_breakdown'][item_name]['location_breakdown'][location_name]['quantity'] += quantity
                results['item_breakdown'][item_name]['location_breakdown'][location_name]['sales'] += item_total
        
        return results

# Flask app configuration
app = Flask(__name__)
app.secret_key = os.urandom(24)

# MSAL app instance
msal_app = ConfidentialClientApplication(
    CLIENT_ID, authority=AUTHORITY, client_credential=CLIENT_SECRET
)

# Global chatbot instance
chatbot = IntegratedDocumentChatbot(OPENAI_API_KEY)

# Initialize database connection
def init_db():
    conn = sqlite3.connect("onedrive_files.db")
    cursor = conn.cursor()
    # Create a table to store file metadata
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS files (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            content TEXT,
            last_modified TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# Route: Home page
@app.route("/")
def index():
    return render_template("index.html")

# Route: Login and redirect to Microsoft's OAuth flow
@app.route("/login")
def login():
    auth_url = msal_app.get_authorization_request_url(SCOPES, redirect_uri=REDIRECT_URI)
    return redirect(auth_url)

# Route: Handle the redirect from Microsoft OAuth
@app.route("/getAToken")
def get_token():
    code = request.args.get("code")
    if not code:
        return "Error: No code provided", 400

    result = msal_app.acquire_token_by_authorization_code(code, SCOPES, redirect_uri=REDIRECT_URI)
    if "access_token" in result:
        session["access_token"] = result["access_token"]
        return redirect("/dashboard")
    else:
        return "Error: Could not acquire token", 400

# Route: Dashboard
@app.route("/dashboard")
def dashboard():
    access_token = session.get("access_token")
    if not access_token:
        return redirect("/login")

    response = requests.get(f"{GRAPH_API_ENDPOINT}/me", headers={"Authorization": f"Bearer {access_token}"})
    user_info = response.json()

    return render_template("dashboard.html", user_name=user_info.get("displayName", "User"))

# Route: Sync files from OneDrive to database
@app.route("/sync_files")
def sync_files():
    access_token = session.get("access_token")
    if not access_token:
        return redirect("/login")

    response = requests.get(f"{GRAPH_API_ENDPOINT}/me/drive/root/children", headers={"Authorization": f"Bearer {access_token}"})
    files = response.json().get("value", [])

    conn = sqlite3.connect("onedrive_files.db")
    cursor = conn.cursor()

    for file in files:
        file_id = file["id"]
        file_name = file["name"]
        file_content_response = requests.get(
            f"{GRAPH_API_ENDPOINT}/me/drive/items/{file_id}/content",
            headers={"Authorization": f"Bearer {access_token}"}
        )

        if file_content_response.status_code == 200:
            # Detect file type
            mime_type, _ = mimetypes.guess_type(file_name)
            file_content = None

            # Handle PDFs
            if mime_type == "application/pdf":
                pdf_reader = PdfReader(io.BytesIO(file_content_response.content))
                file_content = "\n".join(page.extract_text() for page in pdf_reader.pages)

            # Handle Excel files
            elif mime_type in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
                try:
                    # Read all sheets and combine them
                    excel_file = pd.ExcelFile(io.BytesIO(file_content_response.content))
                    all_sheets_content = []
                    
                    for sheet_name in excel_file.sheet_names:
                        df = excel_file.parse(sheet_name)
                        all_sheets_content.append(f"Sheet: {sheet_name}\n")
                        all_sheets_content.append(df.to_string())
                        all_sheets_content.append("\n\n")
                    
                    file_content = "\n".join(all_sheets_content)
                except Exception as e:
                    file_content = f"Error processing Excel file: {str(e)}"

            # Handle other file types as plain text
            else:
                file_content = file_content_response.text

            cursor.execute("""
                INSERT OR REPLACE INTO files (id, name, content, last_modified)
                VALUES (?, ?, ?, ?)
            """, (file_id, file_name, file_content, file.get("lastModifiedDateTime")))

    conn.commit()
    conn.close()

    return render_template("sync_complete.html", file_count=len(files))

# Route: Search files in database
@app.route("/search_files", methods=["GET", "POST"])
def search_files():
    if request.method == "GET":
        return render_template("search_files.html")

    query = request.form["query"]
    conn = sqlite3.connect("onedrive_files.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT name, content FROM files WHERE content LIKE ? OR name LIKE ?
    """, (f"%{query}%", f"%{query}%"))
    results = cursor.fetchall()
    conn.close()

    return render_template("search_results.html", results=results)

@app.route("/chatbot", methods=["GET", "POST"])
def chatbot_interface():
    # Initialize chat history in session if not exists
    if 'chat_history' not in session:
        session['chat_history'] = []
    
    if request.method == "POST":
        # Handle file uploads
        if 'pdf' in request.files:
            pdf_file = request.files['pdf']
            if pdf_file:
                chatbot.load_pdf(pdf_file)
                chatbot.update_vector_store_if_needed()
                return "File uploaded successfully"
        
        # Handle chat question
        question = request.form.get('question')
        if question:
            # Get current chat history
            chat_history = session.get('chat_history', [])
            
            # Generate response
            response = chatbot.answer_question(question, chat_history)
            
            # Update chat history with the new exchange
            chat_history.append({"role": "user", "content": question})
            chat_history.append({"role": "assistant", "content": response})
            
            # Trim chat history if it gets too long (keep last 20 exchanges)
            if len(chat_history) > 40:  # 20 exchanges (40 messages)
                chat_history = chat_history[-40:]
            
            # Save updated history back to session
            session['chat_history'] = chat_history
            
            return render_template(
            "chatbot.html",
            chat_history=chat_history,
            bot_response=response
        )

    # For GET requests, render the template with current chat history
    return render_template(
        "chatbot.html",
        chat_history=session.get('chat_history', [])
    )

# Manual file upload route
@app.route("/upload_files", methods=["GET", "POST"])
def upload_files():
    if request.method == "POST":
        # Handle PDF upload
        if 'pdf' in request.files:
            pdf_file = request.files['pdf']
            if pdf_file.filename != '':
                chatbot.load_pdf(pdf_file)
                chatbot.update_vector_store_if_needed()

        # Handle CSV upload
        if 'csv' in request.files:
            csv_file = request.files['csv']
            if csv_file.filename != '':
                chatbot.load_csv(csv_file)
                chatbot.update_vector_store_if_needed()

        # Handle Excel upload
        if 'excel' in request.files:
            excel_file = request.files['excel']
            if excel_file.filename != '':
                chatbot.load_excel(excel_file)
                chatbot.update_vector_store_if_needed()

        return redirect("/chatbot")

    return render_template("upload_files.html")


@app.route("/clear_chat")
def clear_chat():
    """
    Clear chat history while preserving authentication tokens
    """
    # Store authentication tokens
    ms_token = session.get('access_token')
    square_token = session.get('square_access_token')
    
    # Clear chat history
    session['chat_history'] = []
    
    # Restore authentication tokens
    if ms_token:
        session['access_token'] = ms_token
    if square_token:
        session['square_access_token'] = square_token
    
    return redirect("/chatbot")

@app.route("/reset_database")
def reset_database():
    """
    Reset the OneDrive files database while preserving authentication tokens:
    1. Save existing authentication tokens
    2. Closing any existing connections
    3. Deleting the existing database file
    4. Reinitializing the database
    5. Restore authentication tokens
    """
    # Path to the database file
    db_path = "onedrive_files.db"

    try:
        # Save authentication tokens
        ms_token = session.get('access_token')
        square_token = session.get('square_access_token')

        # Delete the existing database file if it exists
        if os.path.exists(db_path):
            os.remove(db_path)

        # Reinitialize the database
        init_db()

        # Reset chatbot state without affecting authentication
        if 'chatbot' in globals():
            chatbot.manually_uploaded_documents = []
            chatbot.loaded_documents = []
            chatbot.vector_store = None
            chatbot.last_document_count = 0

        # Clear session but preserve authentication tokens
        session.clear()
        if ms_token:
            session['access_token'] = ms_token
        if square_token:
            session['square_access_token'] = square_token

        return "Database reset successfully. Authentication tokens preserved. <a href='/dashboard'>Return to Dashboard</a>"
    except Exception as e:
        return f"Error resetting database: {str(e)}"
    
@app.route("/list_files")
def list_files():
    """
    Retrieve and display all files stored in the database
    """
    show_content = request.args.get('show_content', 'false')
    
    conn = sqlite3.connect("onedrive_files.db")
    cursor = conn.cursor()
    
    if show_content == 'true':
        # Retrieve all files with full content
        cursor.execute("SELECT id, name, content, last_modified FROM files")
    else:
        # Retrieve files with just metadata
        cursor.execute("SELECT id, name, length(content) as content_length, last_modified FROM files")
    
    files = cursor.fetchall()
    
    conn.close()

    return render_template("list_files.html", files=files, show_content=show_content)

@app.route("/connect_square")
def connect_square():
    """
    Initiate Square OAuth flow
    """
    auth_url = f"https://connect.squareup.com/oauth2/authorize?client_id={SQUARE_APPLICATION_ID}"
    auth_url += f"&scope=ORDERS_READ ITEMS_READ MERCHANT_PROFILE_READ"
    auth_url += f"&redirect_uri={SQUARE_REDIRECT_URI}"
    return redirect(auth_url)

@app.route("/squareauth")
def square_callback():
    """
    Handle Square OAuth callback and token exchange
    """
    if 'error' in request.args:
        return f"Error: {request.args.get('error')}"

    code = request.args.get('code')
    if not code:
        return "Error: No authorization code received"

    # Exchange authorization code for access token
    client = Client(
        environment='production'  # Change to 'production' for live data
    )

    body = {
        'client_id': SQUARE_APPLICATION_ID,
        'client_secret': SQUARE_APPLICATION_SECRET,
        'code': code,
        'grant_type': 'authorization_code',
        'redirect_uri': SQUARE_REDIRECT_URI  # Add this line
    }

    response = client.o_auth.obtain_token(body)
    
    if response.is_success():
        # Store the access token in session
        result = response.body
        session['square_access_token'] = result['access_token']
        
        # Initialize Square client with the new token
        return redirect('/chatbot')
    else:
        return f"Error obtaining token: {response.errors}"

# @app.route("/sync_square_data")
# def sync_square_data():
#     """
#     Optional data sync function that caches recent Square data locally.
#     Useful for reducing API calls and maintaining historical data.
#     """
#     access_token = session.get('square_access_token')
#     if not access_token:
#         return redirect('/connect_square')

#     client = Client(access_token=access_token, environment='production')

#     try:
#         # Sync last 7 days of data instead of 30
#         end_time = datetime.utcnow()
#         start_time = end_time - timedelta(days=7)

#         # Query all locations
#         body = {
#             'location_ids': list(LOCATION_IDS.values()),
#             'query': {
#                 'filter': {
#                     'date_time_filter': {
#                         'created_at': {
#                             'start_at': start_time.isoformat(),
#                             'end_at': end_time.isoformat()
#                         }
#                     }
#                 }
#             }
#         }

#         result = client.orders.search_orders(body=body)
        
#         if result.is_success():
#             orders = result.body.get('orders', [])
            
#             conn = sqlite3.connect("onedrive_files.db")
#             cursor = conn.cursor()

#             # Store orders with more complete data for better context
#             for order in orders:
#                 order_id = order['id']
#                 location_id = order.get('location_id')
                
#                 # Get location name from ID
#                 location_name = next(
#                     (name for name, id in LOCATION_IDS.items() if id == location_id),
#                     'unknown location'
#                 )

#                 # Extract items with more detail
#                 items_data = []
#                 for item in order.get('line_items', []):
#                     item_data = {
#                         'name': item.get('name', 'Unknown Item'),
#                         'quantity': int(item.get('quantity', 1)),
#                         'price': float(item.get('total_money', {}).get('amount', 0)) / 100,
#                         'modifiers': [mod['name'] for mod in item.get('modifiers', [])]
#                     }
#                     items_data.append(item_data)

#                 # Create enriched order summary
#                 order_summary = {
#                     'order_id': order_id,
#                     'location_id': location_id,
#                     'location_name': location_name,
#                     'created_at': order.get('created_at'),
#                     'total_money': float(order['total_money']['amount']) / 100,
#                     'items': items_data,
#                     'state': order.get('state'),
#                     'source': order.get('source', {}).get('name', 'unknown')
#                 }

#                 cursor.execute("""
#                     INSERT OR REPLACE INTO files (id, name, content, last_modified)
#                     VALUES (?, ?, ?, ?)
#                 """, (
#                     f"square_order_{order_id}",
#                     f"Square Order {order_id} - {location_name}",
#                     json.dumps(order_summary, indent=2),
#                     order.get('created_at')
#                 ))

#             conn.commit()
#             conn.close()

#             return f"Successfully cached {len(orders)} recent orders. <a href='/dashboard'>Return to Dashboard</a>"
#         else:
#             return f"Error retrieving orders: {result.errors}"

#     except Exception as e:
#         return f"Error syncing Square data: {str(e)}"


# Main application entry point
if __name__ == "__main__":
    # Ensure required directories and files exist
    os.makedirs("templates", exist_ok=True)
    
    # Run the Flask application
    app.run(debug=True, host='0.0.0.0', port=5001)