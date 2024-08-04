import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from requests.exceptions import HTTPError
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def load_document(file):
    if file.type == "application/pdf":
        loader = PyPDFLoader(file)
        documents = loader.load()
    else:
        document = file.read().decode("utf-8")
        documents = [{"text": document}]
    return documents

# Initialize Streamlit
st.title("Document QA System")
st.write("Upload a document (PDF or TXT) and ask questions about it.")

# Upload document
uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt"])

if uploaded_file is not None:
    try:
        # Load document
        documents = load_document(uploaded_file)

        # Split the document into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
        docs = text_splitter.split_documents(documents=documents)

        # Load the embedding model from HuggingFace
        embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

        # Load the data and corresponding embeddings into the FAISS
        vectorstore = FAISS.from_documents(docs, embeddings)

        # Persist the vectors locally on disk
        vectorstore.save_local("faiss_index_")

        # Load from local storage
        persisted_vectorstore = FAISS.load_local("faiss_index_", embeddings, allow_dangerous_deserialization=True)

        # Create a retriever on top of the database
        retriever = persisted_vectorstore.as_retriever()

        # Initialize an instance of the Ollama model
        llm = Ollama(model="llama3.1")

        # Use RetrievalQA chain for orchestration
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        
        st.write("Document loaded and processed. You can now ask questions about it.")

        # Input query from user
        query = st.text_input("Type your query:")

        if query:
            result = qa.run(query)
            st.write(result)
    except HTTPError as http_err:
        st.error(f"HTTP error occurred: {http_err}")
    except Exception as err:
        st.error(f"An error occurred: {err}")
else:
    st.write("Please upload a PDF or text file to proceed.")

# Running the Streamlit app: save this script as `app.py` and run it using the command `streamlit run app.py`
