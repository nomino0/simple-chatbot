import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS

grock_api_key = os.getenv('GROQ_API_KEY')

def load_documents(file_path):
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    return loader.load()

# chunking the documents
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=30
    )
    return splitter.split_documents(documents)

def vector_db(chunks):

    embedder = HuggingFaceBgeEmbeddings('all-MiniLM-L6-v2')
    vdb = FAISS.from_documents(chunks, embedder)
    return vdb

def use_rag(vector_db, question):
    retriever = vector_db.as_retriever()
    llm = ChatGroq(api_key=grock_api_key, model="llama-3.3-70b-versatile")
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    result = chain({"query": question})
    return qa_chain({"query": question})
