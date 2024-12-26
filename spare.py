import os
import requests
import sqlite3
import mimetypes
import io
from flask import Flask, request, redirect, session, render_template
from msal import ConfidentialClientApplication
from PyPDF2 import PdfReader
import pandas as pd
import tiktoken
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings, OpenAI as LangchainOpenAI
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, UnstructuredExcelLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

from square.http.auth.o_auth_2 import BearerAuthCredentials
from square.client import Client
import pandas as pd
from datetime import datetime, timedelta
import pytz
from typing import List, Dict, Optional

class SquareIntegration:
    def __init__(self, access_token: str, location_ids: List[str] = None, environment: str = 'sandbox'):
        """
        Initialize Square API client with support for multiple locations
        
        Args:
            access_token (str): Square API access token
            location_ids (List[str]): List of Square location IDs. If None, fetches all locations
            environment (str): 'sandbox' or 'production'
        """
        self.client = Client(
            bearer_auth_credentials=BearerAuthCredentials(access_token=access_token),
            environment=environment
        )
        self.location_ids = location_ids
        self._init_locations()

    def _init_locations(self):
        """
        Initialize location information, either using provided IDs or fetching all locations
        """
        try:
            result = self.client.locations.list_locations()
            if result.is_success():
                all_locations = {
                    loc['id']: {
                        'name': loc['name'],
                        'address': loc.get('address', {}).get('address_line_1', 'N/A'),
                        'timezone': loc.get('timezone', 'UTC')
                    }
                    for loc in result.body['locations']
                }
                
                if self.location_ids is None:
                    # Use all available locations
                    self.location_ids = list(all_locations.keys())
                
                # Filter to only store info for requested locations
                self.locations = {
                    loc_id: info for loc_id, info in all_locations.items()
                    if loc_id in self.location_ids
                }
            else:
                print(f"Error fetching locations: {result.errors}")
                self.locations = {}
                
        except Exception as e:
            print(f"Error initializing locations: {str(e)}")
            self.locations = {}

    def fetch_transactions(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch transactions from all configured locations
        
        Args:
            start_date (datetime): Start date for transaction fetch
            end_date (datetime): End date for transaction fetch
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping location IDs to their transaction data
        """
        if not start_date:
            start_date = datetime.now(pytz.UTC) - timedelta(days=30)
        if not end_date:
            end_date = datetime.now(pytz.UTC)

        location_data = {}
        
        for location_id in self.location_ids:
            try:
                result = self.client.orders.search_orders(
                    body={
                        "location_ids": [location_id],
                        "query": {
                            "filter": {
                                "date_time_filter": {
                                    "created_at": {
                                        "start_at": start_date.isoformat(),
                                        "end_at": end_date.isoformat()
                                    }
                                }
                            }
                        }
                    }
                )

                if result.is_success():
                    df = self._process_transactions(result.body.get('orders', []), location_id)
                    if not df.empty:
                        location_data[location_id] = df
                else:
                    print(f"Error fetching transactions for location {location_id}: {result.errors}")

            except Exception as e:
                print(f"Error in Square API call for location {location_id}: {str(e)}")

        return location_data

    def _process_transactions(self, orders: List[Dict], location_id: str) -> pd.DataFrame:
        """
        Process raw Square API order data into structured format
        
        Args:
            orders (List[Dict]): Raw order data from Square API
            location_id (str): Location ID for the orders
        
        Returns:
            pd.DataFrame: Processed transaction data
        """
        processed_data = []
        location_info = self.locations.get(location_id, {})
        
        for order in orders:
            # Extract basic order information
            order_data = {
                'order_id': order.get('id'),
                'created_at': order.get('created_at'),
                'location_id': location_id,
                'location_name': location_info.get('name', 'Unknown'),
                'location_address': location_info.get('address', 'N/A'),
                'total_money': sum(float(item.get('base_price_money', {}).get('amount', 0)) / 100 
                                 for item in order.get('line_items', [])),
                'status': order.get('status'),
                'items': [],
                'item_count': 0
            }
            
            # Process line items
            for item in order.get('line_items', []):
                order_data['items'].append({
                    'name': item.get('name'),
                    'quantity': item.get('quantity'),
                    'base_price': float(item.get('base_price_money', {}).get('amount', 0)) / 100
                })
                order_data['item_count'] += int(item.get('quantity', 0))
            
            processed_data.append(order_data)

        return pd.DataFrame(processed_data)

    def get_sales_summary(self, location_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Generate comprehensive sales summary across all locations
        
        Args:
            location_data (Dict[str, pd.DataFrame]): Dictionary of location transaction data
        
        Returns:
            Dict: Multi-location sales summary statistics
        """
        overall_summary = {
            'total_sales': 0,
            'total_orders': 0,
            'total_items_sold': 0,
            'average_order_value': 0,
            'location_summaries': {},
            'sales_by_day': {}
        }
        
        all_transactions = pd.concat(location_data.values(), ignore_index=True) if location_data else pd.DataFrame()
        
        if not all_transactions.empty:
            # Overall statistics
            overall_summary.update({
                'total_sales': all_transactions['total_money'].sum(),
                'total_orders': len(all_transactions),
                'total_items_sold': all_transactions['item_count'].sum(),
                'average_order_value': all_transactions['total_money'].mean(),
            })
            
            # Per-location statistics
            for location_id, df in location_data.items():
                location_name = self.locations[location_id]['name']
                overall_summary['location_summaries'][location_id] = {
                    'name': location_name,
                    'total_sales': df['total_money'].sum(),
                    'total_orders': len(df),
                    'total_items_sold': df['item_count'].sum(),
                    'average_order_value': df['total_money'].mean(),
                    'sales_by_day': df.groupby(pd.to_datetime(df['created_at']).dt.date)['total_money'].sum().to_dict()
                }
            
            # Overall daily sales
            overall_summary['sales_by_day'] = all_transactions.groupby(
                pd.to_datetime(all_transactions['created_at']).dt.date
            )['total_money'].sum().to_dict()
        
        return overall_summary

class IntegratedDocumentChatbot:
    def __init__(self, openai_api_key: str, square_access_token: Optional[str] = None, 
                 square_location_ids: Optional[List[str]] = None, db_path: str = "onedrive_files.db"):
        """
        Initialize the chatbot with OpenAI API key and database connection
        """
        self.openai_api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key

        # Initialize Square integration if credentials provided
        self.square = None
        if square_access_token:
            self.square = SquareIntegration(
                access_token=square_access_token,
                location_ids=square_location_ids
            )

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

        # Store Square data
        self.square_data = {}  # Dictionary mapping location_id to DataFrame
        self.last_square_update = None

    def _convert_square_data_to_docs(self):
        """
        Convert Square transaction data to document format for all locations
        """
        if not self.square_data:
            return []
            
        docs = []
        
        # Create overall summary document
        summary = self.square.get_sales_summary(self.square_data)
        summary_text = f"""Overall Square Sales Summary:
        Total Sales (All Locations): ${summary['total_sales']:,.2f}
        Total Orders: {summary['total_orders']}
        Total Items Sold: {summary['total_items_sold']}
        Average Order Value: ${summary['average_order_value']:,.2f}
        """
        
        docs.append(type('Document', (), {
            'page_content': summary_text,
            'metadata': {'source': 'square_overall_summary'}
        }))
        
        # Create location-specific summary documents
        for loc_id, loc_summary in summary['location_summaries'].items():
            loc_text = f"""Location Summary for {loc_summary['name']}:
            Total Sales: ${loc_summary['total_sales']:,.2f}
            Total Orders: {loc_summary['total_orders']}
            Total Items Sold: {loc_summary['total_items_sold']}
            Average Order Value: ${loc_summary['average_order_value']:,.2f}
            """
            
            docs.append(type('Document', (), {
                'page_content': loc_text,
                'metadata': {'source': f'square_location_{loc_id}'}
            }))
            
            # Add daily sales for each location
            for date, sales in loc_summary['sales_by_day'].items():
                daily_text = f"Daily Sales for {loc_summary['name']} on {date}: ${sales:,.2f}"
                docs.append(type('Document', (), {
                    'page_content': daily_text,
                    'metadata': {'source': f'square_daily_{loc_id}_{date}'}
                }))
        
        return docs

    def update_square_data(self):
        """
        Fetch and update Square transaction data for all locations
        """
        if not self.square:
            return
            
        current_time = datetime.now()
        if (not self.last_square_update or 
            (current_time - self.last_square_update).total_seconds() > 3600):
            
            # Fetch new data for all locations
            self.square_data = self.square.fetch_transactions()
            self.last_square_update = current_time
            
            # Convert Square data to document format
            square_docs = self._convert_square_data_to_docs()
            
            # Remove old Square documents
            self.loaded_documents = [doc for doc in self.loaded_documents 
                                   if not doc.metadata['source'].startswith('square_')]
            
            # Add new Square documents
            self.loaded_documents.extend(square_docs)
            
            # Update vector store
            self.update_vector_store_if_needed()

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
        Answer a question based on loaded documents using OpenAI's ChatCompletion API.
        """
        # Update vector store efficiently
        self.update_vector_store_if_needed()

        # Initialize context for the OpenAI ChatCompletion API
        context = []

        if self.vector_store:
            # Perform similarity search to find relevant chunks
            docs = self.vector_store.similarity_search(question, k=5)

            # Validate token limits for retrieved documents
            docs = self.validate_token_limit(docs, question)

            # Combine document content into a single string
            document_context = "\n\n".join([doc.page_content for doc in docs])

            # Add the document context to the conversation
            context.append({
                "role": "system",
                "content": f"Relevant information from documents:\n{document_context}"
            })

        # Combine chat history with the new context
        full_conversation = context + chat_history + [{"role": "user", "content": question}]

        # Generate response using OpenAI's ChatCompletion API
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=full_conversation
        )

        # Extract the assistant's reply
        answer = response.choices[0].message.content

        return answer

# Flask app configuration
app = Flask(__name__)
app.secret_key = os.urandom(24)

# App configuration (replace these with your own values)
CLIENT_ID = "8c3d5655-72bd-4c68-83a8-5f3ed8e5dd64"
CLIENT_SECRET = "Jmp8Q~s.Cm5nKx7gOVLnUL~Q9SgyF_c~PekQ8aaV"
AUTHORITY = "https://login.microsoftonline.com/common"
REDIRECT_URI = "http://localhost:5001/getAToken"
SCOPES = ["Files.ReadWrite.All", "User.Read"]
GRAPH_API_ENDPOINT = "https://graph.microsoft.com/v1.0"
OPENAI_API_KEY = "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" 
SQUARE_ACCESS_TOKEN = "EAAAl8d8l6l_Eyo-d25rkJdbLfd8_3CPgrnnUqfVzbvS-K-EhxHpU5a-1COSNbi3"
SQUARE_LOCATION_IDS = ["LJZ754B8QFEVZ"]  # Or None to fetch all locations
SQUARE_ENVIRONMENT = "sandbox"  # or "production"


# MSAL app instance
msal_app = ConfidentialClientApplication(
    CLIENT_ID, authority=AUTHORITY, client_credential=CLIENT_SECRET
)

# Global chatbot instance
chatbot = IntegratedDocumentChatbot(
    OPENAI_API_KEY,
    square_access_token=SQUARE_ACCESS_TOKEN,
    square_location_ids=SQUARE_LOCATION_IDS
)

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

# Route: Chatbot interface
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

        if 'csv' in request.files:
            csv_file = request.files['csv']
            if csv_file:
                chatbot.load_csv(csv_file)
                chatbot.update_vector_store_if_needed()

        if 'excel' in request.files:
            excel_file = request.files['excel']
            if excel_file:
                chatbot.load_excel(excel_file)
                chatbot.update_vector_store_if_needed()

        # Handle chat question
        question = request.form.get('question')
        if question:
            # Add user's question to chat history
            session['chat_history'].append({'role': 'user', 'content': question})

            # Generate response
            response = chatbot.answer_question(question# (Previous code continues...)

            , session['chat_history'])

            # Add assistant's response to chat history
            session['chat_history'].append({'role': 'assistant', 'content': response})

    return render_template("chatbot.html", 
                           chat_history=session.get('chat_history', []))

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

# Route to clear chat history
@app.route("/clear_chat")
def clear_chat():
    session['chat_history'] = []
    return redirect("/chatbot")

@app.route("/reset_database")
def reset_database():
    """
    Reset the OneDrive files database by:
    1. Closing any existing connections
    2. Deleting the existing database file
    3. Reinitializing the database
    """

    # Path to the database file
    db_path = "onedrive_files.db"

    try:
        # Delete the existing database file if it exists
        if os.path.exists(db_path):
            os.remove(db_path)

        # Reinitialize the database
        init_db()

        # Clear manually uploaded documents in the chatbot
        if 'chatbot' in globals():
            chatbot.manually_uploaded_documents = []
            chatbot.loaded_documents = []
            chatbot.vector_store = None
            chatbot.last_document_count = 0

        # Clear session data if needed
        session.clear()

        return "Database reset successfully. <a href='/dashboard'>Return to Dashboard</a>"
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

@app.route("/refresh_square_data")
def refresh_square_data():
    chatbot.update_square_data()
    return redirect("/chatbot")

@app.route("/square_status")
def square_status():
    """
    Display the current status of Square integration and data
    """
    if not chatbot.square:
        return "Square integration is not initialized. Check your credentials."
        
    status_data = {
        'locations': chatbot.square.locations,
        'last_update': chatbot.last_square_update,
        'has_data': bool(chatbot.square_data),
        'data_summary': {}
    }
    
    # Add data counts per location
    if chatbot.square_data:
        for loc_id, df in chatbot.square_data.items():
            loc_name = chatbot.square.locations.get(loc_id, {}).get('name', 'Unknown')
            status_data['data_summary'][loc_id] = {
                'location_name': loc_name,
                'transaction_count': len(df),
                'date_range': f"{df['created_at'].min()} to {df['created_at'].max()}" if not df.empty else 'No data'
            }
    
    # Add a preview key to pass to the template
    preview = {
        loc_id: {
            'location_name': chatbot.square.locations.get(loc_id, {}).get('name', 'Unknown'),
            'sample_data': df.head(5).to_dict('records') if not df.empty else [],
            'total_records': len(df)
        }
        for loc_id, df in chatbot.square_data.items()
    } if chatbot.square_data else {}

    return render_template("square_status.html", status=status_data, preview=preview)


@app.route("/force_square_update")
def force_square_update():
    """
    Force an immediate update of Square data
    """
    try:
        chatbot.last_square_update = None  # Reset the last update time
        chatbot.update_square_data()  # Force update
        return redirect("/square_status")
    except Exception as e:
        return f"Error updating Square data: {str(e)}"

@app.route("/square_data_preview")
def square_data_preview():
    """
    Show a preview of the actual Square data
    """
    print("Square data preview called.")
    if not chatbot.square_data:
        print("No Square data available.")
        return "No Square data available."
        
    preview_data = {}
    for loc_id, df in chatbot.square_data.items():
        loc_name = chatbot.square.locations.get(loc_id, {}).get('name', 'Unknown')
        print(f"Location: {loc_name}, Records: {len(df)}")
        preview_data[loc_id] = {
            'location_name': loc_name,
            'sample_data': df.head(5).to_dict('records') if not df.empty else [],
            'total_records': len(df)
        }

    return render_template("square_preview.html", preview=preview_data)


# Update create_templates() to add list_files.html and update dashboard
def create_templates():
    os.makedirs("templates", exist_ok=True)

    # index.html
    with open("templates/index.html", "w") as f:
        f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Rebels N Misfits Chatbot</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container mt-5">
        <h1 class="text-center mb-4">Rebels N Misfits Chatbot</h1>
        <div class="text-center">
            <a href="/login" class="btn btn-primary btn-lg">Login with Microsoft</a>
        </div>
    </div>
</body>
</html>
        ''')

    # dashboard.html
    with open("templates/dashboard.html", "w") as f:
        f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container mt-5">
        <h1>Welcome, {{ user_name }}!</h1>
        <div class="row">
            <div class="col-md-3 mb-3">
                <a href="/sync_files" class="btn btn-primary w-100">Sync OneDrive Files</a>
            </div>
            <div class="col-md-3 mb-3">
                <a href="/search_files" class="btn btn-secondary w-100">Search Files</a>
            </div>
            <div class="col-md-3 mb-3">
                <a href="/chatbot" class="btn btn-success w-100">Rebels N Misfits Chatbot</a>
            </div>
            <div class="col-md-3 mb-3">
                <a href="/reset_database" class="btn btn-danger w-100" onclick="return confirm('Are you sure you want to reset the database? This will delete all synced and uploaded files.');">Reset Database</a>
            </div>
            <div class="col-md-3 mb-3">
                <a href="/list_files" class="btn btn-info w-100">View Stored Files</a>
            </div>
            <div class="col-md-3 mb-3">
                <a href="/refresh_square_data" class="btn btn-info w-100">Refresh Square Data</a>
            </div>
            <div class="col-md-3 mb-3">
                <a href="/square_status" class="btn btn-info w-100">Square Integration Status</a>
            </div>
        </div>
    </div>
</body>
</html>
        ''')

    # list_files.html
    with open("templates/list_files.html", "w") as f:
        f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Stored Files</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        pre {
            max-height: 200px;
            overflow-y: auto;
        }
    </style>
</head>
<body>
    <div class="container mt-5">
        <h1>Stored Files</h1>
        {% if show_content == 'true' %}
            <table class="table table-striped">
                <thead>
                    <tr>
                        <th>File ID</th>
                        <th>File Name</th>
                        <th>Content</th>
                        <th>Last Modified</th>
                    </tr>
                </thead>
                <tbody>
                    {% for file in files %}
                    <tr>
                        <td>{{ file[0] }}</td>
                        <td>{{ file[1] }}</td>
                        <td><pre>{{ file[2][:500] }}{% if file[2]|length > 500 %}... (truncated){% endif %}</pre></td>
                        <td>{{ file[3] or 'N/A' }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            <a href="/list_files" class="btn btn-secondary">Hide Content</a>
        {% else %}
            <table class="table table-striped">
                <thead>
                    <tr>
                        <th>File ID</th>
                        <th>File Name</th>
                        <th>Content Length</th>
                        <th>Last Modified</th>
                    </tr>
                </thead>
                <tbody>
                    {% for file in files %}
                    <tr>
                        <td>{{ file[0] }}</td>
                        <td>{{ file[1] }}</td>
                        <td>{{ file[2] }} characters</td>
                        <td>{{ file[3] or 'N/A' }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            <a href="/list_files?show_content=true" class="btn btn-primary">Show File Contents</a>
        {% endif %}
        <a href="/dashboard" class="btn btn-secondary mt-2">Back to Dashboard</a>
    </div>
</body>
</html>
        ''')
    # sync_complete.html
    with open("templates/sync_complete.html", "w") as f:
        f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Sync Complete</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container mt-5 text-center">
        <h1>Sync Complete!</h1>
        <p>Successfully synced {{ file_count }} files from OneDrive</p>
        <a href="/dashboard" class="btn btn-primary">Back to Dashboard</a>
    </div>
</body>
</html>
        ''')

    # search_files.html
    with open("templates/search_files.html", "w") as f:
        f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Search Files</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container mt-5">
        <h1>Search Files in OneDrive</h1>
        <form action="/search_files" method="post">
            <div class="mb-3">
                <input type="text" class="form-control" id="query" name="query" placeholder="Enter search term" required>
            </div>
            <button type="submit" class="btn btn-primary">Search</button>
            <a href="/dashboard" class="btn btn-secondary">Back to Dashboard</a>
        </form>
    </div>
</body>
</html>
        ''')

    # search_results.html
    with open("templates/search_results.html", "w") as f:
        f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Search Results</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container mt-5">
        <h1>Search Results</h1>
        {% if results %}
            <ul class="list-group">
            {% for name, content in results %}
                <li class="list-group-item">
                    <strong>{{ name }}</strong>
                    <p>{{ content[:300] }}...</p>
                </li>
            {% endfor %}
            </ul>
        {% else %}
            <p>No results found.</p>
        {% endif %}
        <a href="/dashboard" class="btn btn-primary mt-3">Back to Dashboard</a>
    </div>
</body>
</html>
        ''')

    # chatbot.html
    with open("templates/chatbot.html", "w") as f:
        f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Document Chatbot</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .chat-container {
            max-height: 500px;
            overflow-y: auto;
        }
    </style>
</head>
<body>
    <div class="container mt-5">
        <h1>Document Chatbot</h1>
        
        <!-- File Upload Section -->
        <div class="row mb-3">
            <div class="col">
                <a href="/upload_files" class="btn btn-primary">Upload Documents</a>
                <a href="/clear_chat" class="btn btn-warning">Clear Chat History</a>
            </div>
        </div>

        <!-- Chat History -->
        <div class="card">
            <div class="card-body chat-container">
                {% for message in chat_history %}
                    {% if message.role == 'user' %}
                        <div class="alert alert-primary text-end">
                            <strong>You:</strong> {{ message.content }}
                        </div>
                    {% else %}
                        <div class="alert alert-secondary">
                            <strong>Assistant:</strong> {{ message.content }}
                        </div>
                    {% endif %}
                {% endfor %}
            </div>
        </div>

        <!-- Chat Input -->
        <form action="/chatbot" method="post" class="mt-3">
            <div class="input-group">
                <input type="text" class="form-control" name="question" placeholder="Ask a question about your documents" required>
                <button type="submit" class="btn btn-success">Send</button>
            </div>
        </form>

        <div class="mt-3">
            <a href="/dashboard" class="btn btn-secondary">Back to Dashboard</a>
        </div>
    </div>
</body>
</html>
        ''')

    # upload_files.html
    with open("templates/upload_files.html", "w") as f:
        f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Upload Files</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container mt-5">
        <h1>Upload Documents</h1>
        <form action="/upload_files" method="post" enctype="multipart/form-data">
            <div class="mb-3">
                <label for="pdf" class="form-label">Upload PDF</label>
                <input class="form-control" type="file" id="pdf" name="pdf" accept=".pdf">
            </div>
            <div class="mb-3">
                <label for="csv" class="form-label">Upload CSV</label>
                <input class="form-control" type="file" id="csv" name="csv" accept=".csv">
            </div>
            <div class="mb-3">
                <label for="excel" class="form-label">Upload Excel</label>
                <input class="form-control" type="file" id="excel" name="excel" accept=".xlsx,.xls">
            </div>
            <button type="submit" class="btn btn-primary">Upload</button>
            <a href="/chatbot" class="btn btn-secondary">Back to Chatbot</a>
        </form>
    </div>
</body>
</html>
        ''')

# Create templates
create_templates()

# Main application entry point
if __name__ == "__main__":
    # Ensure required directories and files exist
    os.makedirs("templates", exist_ok=True)
    
    # Run the Flask application
    app.run(debug=True, host='0.0.0.0', port=5001)