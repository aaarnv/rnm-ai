import os
import streamlit as st
import sqlite3
import requests
import PyPDF2
from langchain_openai import OpenAIEmbeddings, OpenAI as LangchainOpenAI
from langchain.document_loaders import PyPDFLoader, CSVLoader, UnstructuredExcelLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import tiktoken
from openai import OpenAI

class IntegratedDocumentChatbot:
    def __init__(self, openai_api_key, db_path="onedrive_files.db"):
        """
        Initialize the chatbot with OpenAI API key and database connection

        Args:
            openai_api_key (str): OpenAI API key for authentication
            db_path (str): Path to the SQLite database
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

    def load_pdf(self, pdf_file):
        """
        Load PDF file and split into chunks for manual upload

        Args:
            pdf_file (file): PDF file to load
        """
        with open("temp.pdf", "wb") as f:
            f.write(pdf_file.getvalue())

        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        
        # Add to manually uploaded documents
        self.manually_uploaded_documents.extend(texts)

    def load_csv(self, csv_file):
        """
        Load CSV file for manual upload

        Args:
            csv_file (file): CSV file to load
        """
        with open("temp.csv", "wb") as f:
            f.write(csv_file.getvalue())

        loader = CSVLoader(file_path="temp.csv")
        documents = loader.load()
        
        # Add to manually uploaded documents
        self.manually_uploaded_documents.extend(documents)

    def load_excel(self, excel_file):
        """
        Load Excel file for manual upload

        Args:
            excel_file (file): Excel file to load
        """
        with open("temp.xlsx", "wb") as f:
            f.write(excel_file.getvalue())

        loader = UnstructuredExcelLoader("temp.xlsx")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        
        # Add to manually uploaded documents
        self.manually_uploaded_documents.extend(texts)

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
                self.vector_store = FAISS.from_documents(
                    self.loaded_documents, 
                    self.embeddings
                )
                
                # Update tracking
                self.last_document_count = current_doc_count

    def validate_token_limit(self, docs, question):
        """
        Validate if the token limit is exceeded.

        Args:
            docs (list): List of document chunks
            question (str): User's question

        Returns:
            list: Filtered documents within token limit
        """
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        question_tokens = len(encoding.encode(question))
        doc_tokens = [len(encoding.encode(doc.page_content)) for doc in docs]

        total_tokens = question_tokens
        valid_docs = []

        for tokens, doc in zip(doc_tokens, docs):
            if total_tokens + tokens <= 4097 - 256:  # Reserve space for output
                total_tokens += tokens
                valid_docs.append(doc)
            else:
                break

        return valid_docs

    def answer_question(self, question):
        """
        Answer a question based on loaded documents using OpenAI's ChatCompletion API.

        Args:
            question (str): User's question.

        Returns:
            str: Generated answer.
        """
        # Ensure the chat history exists in session state
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = [{"role": "system", "content": "You are a helpful assistant analyzing documents."}]

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
        full_conversation = context + st.session_state.chat_history + [{"role": "user", "content": question}]

        # Generate response using OpenAI's ChatCompletion API
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=full_conversation
        )

        # Extract the assistant's reply
        answer = response.choices[0].message.content

        return answer

def main():
    st.title("Integrated Document Chatbot 📁🤖")

    # Sidebar for API Key and File Upload
    st.sidebar.header("Configuration")
    openai_api_key = "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"  # Replace with your actual API key

    # Initialize chatbot
    if openai_api_key:
        chatbot = IntegratedDocumentChatbot(openai_api_key)

        # File upload sections
        uploaded_pdf = st.sidebar.file_uploader("Upload Additional PDF", type="pdf")
        uploaded_csv = st.sidebar.file_uploader("Upload Additional CSV", type="csv")
        uploaded_excel = st.sidebar.file_uploader("Upload Additional Excel", type="xlsx")

        # Load manually uploaded files
        if uploaded_pdf:
            chatbot.load_pdf(uploaded_pdf)
            st.sidebar.success("PDF Uploaded Successfully")

        if uploaded_csv:
            chatbot.load_csv(uploaded_csv)
            st.sidebar.success("CSV Uploaded Successfully")

        if uploaded_excel:
            chatbot.load_excel(uploaded_excel)
            st.sidebar.success("Excel Uploaded Successfully")

        # Chat interface
        st.header("Ask a Question About Your Documents")
        
        # Initialize chat history if it doesn't exist
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        # Display chat history
        for item in st.session_state.chat_history:
            st.markdown(f"**{item['role'].capitalize()}:** {item['content']}")

        user_question = st.text_input("Enter your question about the documents")

        if user_question:
            st.session_state.chat_history.append({'role': 'user', 'content': user_question})

            with st.spinner("Generating response..."):
                response = chatbot.answer_question(user_question)

                # Add both the user's question and the assistant's answer to chat history
                st.session_state.chat_history.append({'role': 'assistant', 'content': response})

                st.write(response)

if __name__ == "__main__":
    main()