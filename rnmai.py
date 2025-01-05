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
        Handle Square queries using chat history for context, with intelligent date handling.
        """
        # Extract date and location information using OpenAI with context from chat history
        context_messages = [
            {
                "role": "system", 
                "content": """You are a JSON generator that specializes in date and location parsing for Square data queries.
                Analyze the current question AND previous context to determine dates and locations.
                For dates mentioned without a year:
                - If the date is in the future for this year, use the previous year
                - If the date is in the past for this year, use this year
                Return dates in YYYY-MM-DD format.
                
                Return a valid JSON object with this structure:
                {
                    "start_date": "YYYY-MM-DD",
                    "end_date": "YYYY-MM-DD",
                    "location": "location_name",
                    "metric": "sales/items",
                    "item_name": null,
                    "relative_date": null,
                    "is_range": false,
                    "from_context": {
                        "date": false,
                        "location": false
                    }
                }
                
                Example:
                If today is January 5, 2025 and user asks about "December 17th":
                - December 17, 2024 has already passed
                - Return "2024-12-17" as the date
                
                If user asks about "January 20th":
                - January 20, 2025 is in the future
                - Return "2024-01-20" as the date
                """
            }
        ]
        
        # Add current date info for context
        current_date = datetime.now(pytz.timezone('Australia/Sydney'))
        context_messages[0]["content"] += f"\nCurrent date: {current_date.strftime('%Y-%m-%d')}"
        
        # Filter chat history to only include Square-related queries
        if chat_history:
            square_context = [
                msg for msg in chat_history 
                if 'square' in msg.get('content', '').lower()
            ]
            if square_context:
                context_messages.extend(square_context[-3:])
        
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
            
            # Parse and adjust dates based on current date
            if query_params.get('is_range'):
                start_time = sydney_tz.localize(datetime.strptime(query_params['start_date'], '%Y-%m-%d'))
                end_time = sydney_tz.localize(datetime.strptime(query_params['end_date'], '%Y-%m-%d'))
                
                # If dates are in future, move them back one year
                if start_time > current_date:
                    start_time = start_time.replace(year=start_time.year - 1)
                if end_time > current_date:
                    end_time = end_time.replace(year=end_time.year - 1)
                    
                end_time = end_time.replace(hour=23, minute=59, second=59)
            else:
                query_date = datetime.strptime(query_params['start_date'], '%Y-%m-%d')
                query_date = sydney_tz.localize(query_date)
                
                # If date is in future, move it back one year
                if query_date > current_date:
                    query_date = query_date.replace(year=query_date.year - 1)
                    
                start_time = query_date.replace(hour=0, minute=0, second=0)
                end_time = query_date.replace(hour=23, minute=59, second=59)

            # Rest of the function remains the same...
            orders = []
            use_cache = self.should_use_cache(start_time, end_time)
            
            if use_cache:
                cached_orders = self.get_cached_orders(start_time, end_time, location_id)
                orders.extend(cached_orders)
            
            if not use_cache or datetime.now(sydney_tz).date() == end_time.date():
                real_time_orders = self.get_real_time_orders(client, start_time, end_time, location_id)
                orders.extend(real_time_orders)
                
            if orders:
                processed_data = self.process_square_orders(
                    orders, 
                    query_params.get('metric'),
                    query_params.get('item_name')
                )
                
                context_info = []
                if query_params['from_context'].get('date'):
                    context_info.append("using date from previous query")
                if query_params['from_context'].get('location'):
                    context_info.append("using location from previous query")
                    
                context_str = f"Analysis for {query_params.get('location', 'all locations')} "
                if query_params.get('is_range'):
                    context_str += f"from {start_time.date()} to {end_time.date()}"
                else:
                    context_str += f"on {start_time.date()}"
                    
                if context_info:
                    context_str += f" ({', '.join(context_info)})"
                    
                context_str += f":\n{json.dumps(processed_data, indent=2)}"
                
                response_messages = [
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant analyzing Square sales data. "
                                "Provide insights based on the data and maintain continuity with previous questions. "
                                "When using context from previous queries, acknowledge this naturally in your response."
                    },
                    {"role": "system", "content": context_str}
                ]
                
                if chat_history:
                    response_messages.extend(chat_history[-3:])
                    
                response_messages.append({"role": "user", "content": question})
                
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=response_messages
                )
                
                return response.choices[0].message.content
            else:
                date_str = f"between {start_time.date()} and {end_time.date()}" if query_params.get('is_range') else f"on {start_time.date()}"
                return f"No orders found for {query_params.get('location', 'the specified location')} {date_str}"
                
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

@app.route("/")
def index():
    return render_template("chatbot.html", chat_history=session.get('chat_history', []))

@app.route("/connect_outlook")
def connect_outlook():
    auth_url = msal_app.get_authorization_request_url(SCOPES, redirect_uri=REDIRECT_URI)
    return redirect(auth_url)

def sync_files_from_onedrive(access_token):
    """Sync files from OneDrive to local database"""
    response = requests.get(
        f"{GRAPH_API_ENDPOINT}/me/drive/root/children", 
        headers={"Authorization": f"Bearer {access_token}"}
    )
    
    if response.status_code != 200:
        raise Exception(f"Failed to fetch files: {response.status_code}")
        
    files = response.json().get("value", [])
    
    conn = sqlite3.connect("onedrive_files.db")
    cursor = conn.cursor()
    
    for file in files:
        file_id = file["id"]
        file_name = file["name"]
        last_modified = file.get("lastModifiedDateTime")
        
        file_content_response = requests.get(
            f"{GRAPH_API_ENDPOINT}/me/drive/items/{file_id}/content",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        
        if file_content_response.status_code == 200:
            mime_type, _ = mimetypes.guess_type(file_name)
            file_content = None
            
            # Handle PDFs
            if mime_type == "application/pdf":
                pdf_reader = PdfReader(io.BytesIO(file_content_response.content))
                file_content = "\n".join(page.extract_text() for page in pdf_reader.pages)
            
            # Handle Excel files
            elif mime_type in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
                try:
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
            """, (file_id, file_name, file_content, last_modified))
    
    conn.commit()
    conn.close()

# Modify the getAToken route to sync immediately after login
@app.route("/getAToken")
def get_token():
    code = request.args.get("code")
    if not code:
        return "Error: No code provided", 400
        
    result = msal_app.acquire_token_by_authorization_code(code, SCOPES, redirect_uri=REDIRECT_URI)
    
    if "access_token" in result:
        session["access_token"] = result["access_token"]
        # Sync files immediately after getting access token
        try:
            sync_files_from_onedrive(result["access_token"])
        except Exception as e:
            print(f"Sync error: {str(e)}")
    
    return redirect("/")

@app.route("/connect_square")
def connect_square():
    auth_url = f"https://connect.squareup.com/oauth2/authorize?client_id={SQUARE_APPLICATION_ID}&scope=ORDERS_READ ITEMS_READ MERCHANT_PROFILE_READ&redirect_uri={SQUARE_REDIRECT_URI}"
    return redirect(auth_url)

@app.route("/squareauth")
def square_callback():
    code = request.args.get('code')
    if not code:
        return "Error: No authorization code received"
    client = Client(environment='production')
    body = {
        'client_id': SQUARE_APPLICATION_ID,
        'client_secret': SQUARE_APPLICATION_SECRET,
        'code': code,
        'grant_type': 'authorization_code',
        'redirect_uri': SQUARE_REDIRECT_URI
    }
    response = client.o_auth.obtain_token(body)
    if response.is_success():
        session['square_access_token'] = response.body['access_token']
    return redirect("/")

@app.route("/upload_files", methods=["POST"])
def upload_files():
    if 'pdf' in request.files:
        pdf_file = request.files['pdf']
        if pdf_file:
            chatbot.load_pdf(pdf_file)
    if 'csv' in request.files:
        csv_file = request.files['csv']
        if csv_file:
            chatbot.load_csv(csv_file)
    if 'excel' in request.files:
        excel_file = request.files['excel']
        if excel_file:
            chatbot.load_excel(excel_file)
    return redirect("/")

@app.route("/chatbot", methods=["POST"])
def chatbot_interface():
    question = request.form.get('question')
    if question:
        chat_history = session.get('chat_history', [])
        response = chatbot.answer_question(question, chat_history)
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": response})
        session['chat_history'] = chat_history[-40:]
        print(response)
        return response
    return "No question provided", 400

@app.route("/clear_chat")
def clear_chat():
    session['chat_history'] = []
    return redirect("/")
    

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)

