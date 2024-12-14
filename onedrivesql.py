import os
import requests
import sqlite3
from flask import Flask, request, redirect, session
from msal import ConfidentialClientApplication
import threading
import time

# Flask app setup
app = Flask(__name__)
app.secret_key = os.urandom(24)

# App configuration (replace these with your own values)
CLIENT_ID = "35ce19d8-f515-4da5-9b49-142d8b751de6"
CLIENT_SECRET = "pZd8Q~zyvfjhFu0t5QdFIgZSOYYNtnrCYmSpIaWn"
AUTHORITY = "https://login.microsoftonline.com/common"
REDIRECT_URI = "http://localhost:5001/getAToken"
SCOPES = ["Files.ReadWrite.All", "User.Read"]
GRAPH_API_ENDPOINT = "https://graph.microsoft.com/v1.0"

# MSAL app instance
msal_app = ConfidentialClientApplication(
    CLIENT_ID, authority=AUTHORITY, client_credential=CLIENT_SECRET
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
    return """
    <h1>Welcome to OneDrive Database Manager</h1>
    <a href="/login">Login with Microsoft</a>
    """

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

    return f"""
    <h1>Welcome {user_info.get("displayName", "User")}</h1>
    <a href="/list_files">List OneDrive Files</a><br>
    <a href="/sync_files">Sync OneDrive Files to Database</a><br>
    <a href="/search_files">Search Files</a>
    """

# Function: Sync files from OneDrive to database
def sync_files():
    access_token = session.get("access_token")
    if not access_token:
        print("Access token not available. Login required.")
        return

    response = requests.get(f"{GRAPH_API_ENDPOINT}/me/drive/root/children", headers={"Authorization": f"Bearer {access_token}"})
    files = response.json().get("value", [])

    conn = sqlite3.connect("onedrive_files.db")
    cursor = conn.cursor()

    for file in files:
        file_id = file["id"]
        file_content_response = requests.get(
            f"{GRAPH_API_ENDPOINT}/me/drive/items/{file_id}/content",
            headers={"Authorization": f"Bearer {access_token}"}
        )

        file_content = file_content_response.text if file_content_response.status_code == 200 else None

        cursor.execute("""
            INSERT OR REPLACE INTO files (id, name, content, last_modified)
            VALUES (?, ?, ?, ?)
        """, (file_id, file["name"], file_content, file.get("lastModifiedDateTime")))

    conn.commit()
    conn.close()
    print(f"Synced {len(files)} files to database!")

# Background thread to automate sync process
def automated_sync(interval=300):
    while True:
        print("Starting automated sync...")
        try:
            sync_files()
        except Exception as e:
            print(f"Error during sync: {e}")
        time.sleep(interval)

# Route: Search files in database
@app.route("/search_files", methods=["GET", "POST"])
def search_files():
    if request.method == "GET":
        return """
        <h1>Search Files in OneDrive</h1>
        <form action="/search_files" method="post">
            <label for="query">Search Query:</label>
            <input type="text" id="query" name="query" required>
            <br>
            <button type="submit">Search</button>
        </form>
        <a href='/dashboard'>Back to Dashboard</a>
        """

    query = request.form["query"]
    conn = sqlite3.connect("onedrive_files.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT name, content FROM files WHERE content LIKE ? OR name LIKE ?
    """, (f"%{query}%", f"%{query}%"))
    results = cursor.fetchall()
    conn.close()

    result_list = "<ul>"
    for name, content in results:
        result_list += f"<li><strong>{name}</strong><br>{content[:200]}...</li>"
    result_list += "</ul>"

    return f"<h1>Search Results</h1>{result_list}<a href='/dashboard'>Back to Dashboard</a>"


@app.route("/query", methods=["POST"])
def query():
    data = request.json
    sql_query = data.get("sql_query")  # SQL query sent from the Assistant
    if not sql_query:
        return {"error": "SQL query is required"}, 400

    try:
        # Execute the SQL query on the SQLite database
        conn = sqlite3.connect("onedrive_files.db")
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        column_names = [description[0] for description in cursor.description]  # Get column names
        conn.close()

        # Return results in a structured format (e.g., CSV-like or JSON array)
        return {
            "results": [dict(zip(column_names, row)) for row in results]
        }, 200
    except sqlite3.Error as e:
        return {"error": f"SQL execution failed: {str(e)}"}, 500

if __name__ == "__main__":
    threading.Thread(target=automated_sync, daemon=True).start()
    app.run(debug=True, host='0.0.0.0', port=5001)
