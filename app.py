import os
import streamlit as st
import pandas as pd
import PyPDF2
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader, CSVLoader, UnstructuredExcelLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

class BusinessDocumentChatbot:
    def __init__(self, openai_api_key):
        """
        Initialize the chatbot with OpenAI API key
        
        Args:
            openai_api_key (str): OpenAI API key for authentication
        """
        self.openai_api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # Initialize OpenAI components
        self.embeddings = OpenAIEmbeddings()
        self.llm = OpenAI(temperature=0.3)
        
        # Store for loaded documents
        self.loaded_documents = []
        
    def load_pdf(self, pdf_file):
        """
        Load PDF file and split into chunks
        
        Args:
            pdf_file (file): PDF file to load
        """
        # Save uploaded file temporarily
        with open("temp.pdf", "wb") as f:
            f.write(pdf_file.getvalue())
        
        # Load PDF
        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        
        self.loaded_documents.extend(texts)
        
    def load_csv(self, csv_file):
        """
        Load CSV file and convert to text chunks
        
        Args:
            csv_file (file): CSV file to load
        """
        # Save uploaded file temporarily
        with open("temp.csv", "wb") as f:
            f.write(csv_file.getvalue())
        
        # Load CSV
        loader = CSVLoader(file_path="temp.csv")
        documents = loader.load()
        
        self.loaded_documents.extend(documents)
        
    def load_excel(self, excel_file):
        """
        Load Excel file and convert to text chunks
        
        Args:
            excel_file (file): Excel file to load
        """
        # Save uploaded file temporarily
        with open("temp.xlsx", "wb") as f:
            f.write(excel_file.getvalue())
        
        # Load Excel
        loader = UnstructuredExcelLoader("temp.xlsx")
        documents = loader.load()
        
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        
        self.loaded_documents.extend(texts)
        
    def create_vector_store(self):
        """
        Create vector store from loaded documents
        
        Returns:
            FAISS: Vector store for semantic search
        """
        if not self.loaded_documents:
            raise ValueError("No documents have been loaded")
        
        return FAISS.from_documents(self.loaded_documents, self.embeddings)
    
    def answer_question(self, question):
        """
        Answer a question based on loaded documents or directly using LLM
        
        Args:
            question (str): User's question
            
        Returns:
            str: Generated answer
        """
        if not self.loaded_documents:
            # Fall back to using the language model directly if no documents are loaded
            response = self.llm(question)
            return response
        
        # Create vector store if documents are loaded
        vector_store = self.create_vector_store()
        
        # Perform similarity search
        docs = vector_store.similarity_search(question)
        
        # Load QA chain
        chain = load_qa_chain(self.llm, chain_type="stuff")
        
        # Generate answer
        response = chain.run(input_documents=docs, question=question)
        return response


def main():
    st.title("Rebels N Misfits Bot🤖")
    
    # Sidebar for API Key and File Upload
    st.sidebar.header("Configuration")
    openai_api_key = "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"
    
    # File upload sections
    uploaded_pdf = st.sidebar.file_uploader("Upload PDF", type="pdf")
    uploaded_csv = st.sidebar.file_uploader("Upload CSV", type="csv")
    uploaded_excel = st.sidebar.file_uploader("Upload Excel", type="xlsx")
    
    # Initialize chatbot
    if openai_api_key:
        chatbot = BusinessDocumentChatbot(openai_api_key)
        
        # Load files if uploaded
        if uploaded_pdf:
            chatbot.load_pdf(uploaded_pdf)
            st.sidebar.success("PDF Loaded Successfully")
        
        if uploaded_csv:
            chatbot.load_csv(uploaded_csv)
            st.sidebar.success("CSV Loaded Successfully")
        
        if uploaded_excel:
            chatbot.load_excel(uploaded_excel)
            st.sidebar.success("Excel Loaded Successfully")
        
        # Chat interface
        st.header("Ask a Question")
        user_question = st.text_input("Enter your question about the documents")
        
        if user_question:
            with st.spinner("Generating response..."):
                response = chatbot.answer_question(user_question)
                st.write(response)

if __name__ == "__main__":
    main()