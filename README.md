# rnm-ai
https://rnm-ai.onrender.com

# Rebels N Misfits Cafe Chatbot System

## Project Overview

This project is an integrated document management and chatbot system built for Rebels N Misfits Cafe. It combines document processing capabilities with Square payment system integration and natural language processing to provide business insights and answer document-based questions.

![image](https://github.com/user-attachments/assets/85672267-f86b-487e-969b-2124aaa840a5)


## Key Features

1. **Document Management**
   - Supports PDF, CSV, and Excel file uploads
   - OneDrive integration for file synchronization
   - Document chunking and vector storage using FAISS
   - Persistent storage in SQLite database

2. **Square Integration**
   - Real-time sales data retrieval
   - Historical order caching
   - Location-specific analytics
   - Item-level sales tracking

3. **Chatbot Capabilities**
   - Context-aware conversations using OpenAI
   - Document-based question answering
   - Sales data analysis and reporting
   - Intelligent date parsing for Square queries

4. **Authentication**
   - Microsoft Authentication Library (MSAL) for OneDrive
   - Square OAuth2 authentication

## Technology Stack

- **Backend**: Python, Flask
- **AI/ML**: OpenAI API, LangChain, FAISS
- **Database**: SQLite
- **Integrations**: 
  - Square API (Orders, Items, Merchant Profile)
  - Microsoft Graph API (OneDrive)
- **Document Processing**: PyPDF2, pandas
- **Environment**: dotenv for configuration

## System Architecture

### Main Components

1. **IntegratedDocumentChatbot Class**
   - Manages document loading and processing
   - Handles vector store updates
   - Processes Square API queries
   - Manages conversation context

2. **Flask Application**
   - RESTful endpoints for user interaction
   - Session management
   - File upload handling
   - Authentication flows

3. **Database Schema**
   - `files` table: stores document metadata and content
   - Persistent storage for OneDrive and manual uploads

### Data Flow

1. Document Upload → Processing → Vector Store
2. User Query → Context Retrieval → AI Response
3. Square Query → API/Data Cache → Processed Results

## Key Functionalities

### Document Processing
- Supports multiple file formats (PDF, CSV, Excel)
- Automatic text chunking (500 characters, 100 overlap)
- Vector embedding using OpenAI embeddings
- Efficient updates based on document changes

### Square Integration
- Timezone-aware queries (Australia/Sydney)
- Intelligent date handling for past/future references
- Cached and real-time data combination
- Detailed sales and item breakdowns

### Chatbot Features
- Token limit validation (4096 tokens)
- Context maintenance (last 40 messages)
- Document-based answers
- Sales data insights

## Configuration
Required environment variables:
CLIENT_ID
CLIENT_SECRET
AUTHORITY
REDIRECT_URI
GRAPH_API_ENDPOINT
OPENAI_API_KEY
SQUARE_APPLICATION_ID
SQUARE_APPLICATION_SECRET
SQUARE_REDIRECT_URI
LOCATION_IDS

## Usage

1. Start the application:
python app.py

## Web interface: 
Available endpoints:
/connect_outlook (OneDrive auth)
/connect_square (Square auth)
/upload_files (Manual uploads)
/chatbot (Query interface)

## Future Improvements:
- Add multi-user support
- Improve UI to make it more intuitive
- Add caching optimization for Square data
- Implement more advanced document preprocessing
- Enhance error handling and logging
- Add export capabilities for analysis results
