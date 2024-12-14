import os
import streamlit as st
import PyPDF2
from langchain_openai import OpenAIEmbeddings, OpenAI as LangchainOpenAI
from langchain.document_loaders import PyPDFLoader, CSVLoader, UnstructuredExcelLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import tiktoken
from openai import OpenAI

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
        self.llm = LangchainOpenAI(temperature=0.3, max_tokens=100)
        self.client = OpenAI(api_key=openai_api_key)

        # Store for loaded documents
        self.loaded_documents = []
        
    def load_pdf(self, pdf_file):
        """
        Load PDF file and split into chunks

        Args:
            pdf_file (file): PDF file to load
        """
        with open("temp.pdf", "wb") as f:
            f.write(pdf_file.getvalue())

        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        self.loaded_documents.extend(texts)

    def load_csv(self, csv_file):
        """
        Load CSV file and convert to text chunks

        Args:
            csv_file (file): CSV file to load
        """
        with open("temp.csv", "wb") as f:
            f.write(csv_file.getvalue())

        loader = CSVLoader(file_path="temp.csv")
        documents = loader.load()
        self.loaded_documents.extend(documents)

    def load_excel(self, excel_file):
        """
        Load Excel file and convert to text chunks

        Args:
            excel_file (file): Excel file to load
        """
        with open("temp.xlsx", "wb") as f:
            f.write(excel_file.getvalue())

        loader = UnstructuredExcelLoader("temp.xlsx")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
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

    def summarize_documents(self):
        """
        Summarize loaded documents to reduce size.
        """
        summarized_texts = []
        for doc in self.loaded_documents:
            summary = self.llm(f"Summarize this: {doc.page_content}")
            summarized_texts.append(summary)
        self.loaded_documents = summarized_texts

    def answer_question(self, question):
        """
        Answer a question based on loaded documents and chat history using OpenAI's ChatCompletion API.

        Args:
            question (str): User's question.

        Returns:
            str: Generated answer.
        """
        # Ensure the chat history exists in session state
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = [{"role": "system", "content": "You are a helpful assistant."}]

        # Initialize context for the OpenAI ChatCompletion API
        context = []

        if self.loaded_documents:
            # Create vector store from loaded documents
            vector_store = self.create_vector_store()

            # Perform similarity search to find relevant chunks
            docs = vector_store.similarity_search(question, k=5)

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
    st.title("Rebels N Misfits Bot🤖")

    # Sidebar for API Key and File Upload
    st.sidebar.header("Configuration")
    openai_api_key = "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"  # Replace with your actual API key

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

        # Summarize documents to reduce size
        if st.sidebar.button("Summarize Documents"):
            with st.spinner("Summarizing documents..."):
                chatbot.summarize_documents()
                st.sidebar.success("Documents Summarized Successfully")

        # Chat interface
        st.header("Ask a Question")
        
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