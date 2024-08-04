from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA


# Load document using PyPDFLoader document loader
#loader = PyPDFLoader("/home/abdul-qadir/ragwithllama3.1/nvidia-learning-training-course-catalog.pdf")
#documents = loader.load()
with open("/home/abdul-qadir/ragwithllama3.1/Subject Elevate Your YouTube.txt","r") as f:
  document = f.read()



#Splitting the data into chunk
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
docs = text_splitter.split_documents(documents=documents)


#loading the embedding model from huggingface
embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
#model_kwargs = {"device": "cuda"}
embeddings = HuggingFaceEmbeddings(
  model_name=embedding_model_name,
  #model_kwargs=model_kwargs
)

'''
from langchain.vectorstores import FAISS
vectorstore=FAISS.from_documents(text_chunks, embeddings)
retriever=vectorstore.as_retriever()
'''


#loading the data and correspond embedding into the FAISS
vectorstore = FAISS.from_documents(docs, embeddings)


# Persist the vectors locally on disk
vectorstore.save_local("faiss_index_")


# Load from local storage
persisted_vectorstore = FAISS.load_local("faiss_index_", embeddings,allow_dangerous_deserialization=True)
     

#creating a retriever on top of database
retriever = persisted_vectorstore.as_retriever()


from langchain_community.llms import Ollama


# Initialize an instance of the Ollama model
llm = Ollama(model="llama3.1")

# Invoke the model to generate responses
response = llm.invoke("Tell me a joke")
print(response)


#Use RetrievalQA chain for orchestration
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
     
while True:
  query = input("Type your query if you want to exit type Exit: \n")
  if query == "Exit":
    break
  result = qa.run(query)
  print(result)
    