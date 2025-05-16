import os
from groq import Groq
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.huggingface import huggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def pdf_loader(file_path):
    loader= PyPDFLoader(file_path)
    return loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter
def split_document(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30)
    return splitter.split_documents(documents)

def embed_texts(chunks):
    texts = [chunk.page_content for chunk in chunks]
    embedding_model = huggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    embeddings = embedding_model.embed_documents(texts)
    return embeddings


def vector_db(chunks):
    embedding_model = huggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    vdb = FAISS.from_documents(chunks, embedding_model)
    return vdb


def rag(vdb, question):
    retriever = vdb.as_retriever()
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile")
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    result = chain({"query": question})
    return result

def main():
    print("Lecture d'un PDF")
    documents = pdf_loader('knowledge_docs/Radboud_University_Internship_Report_Template__AI_.pdf')
    
    print("Chunking du text")
    chunks= split_document(documents)
    print(type(chunks))
    for chunk in chunks:
        print(chunk)
        break

    print('Créer Vector DB et la replir')
    vdb = vector_db(chunks)

    print('utiliser RAG')
    question = input('Posez Votre Question : ')
    while question != '/bye':
        result = rag(vdb, question)
        print("AI : ", result['result'])
        question = input('\nPosez Votre Question : ')
    else:
        print("\nAI : ", "BYE BYE Don't come back please ")

if _name_ == "_main_":
    main()