import streamlit as st
import os
from dotenv import load_dotenv

# Configure page - MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="AI Assistant", page_icon=":robot_face:", layout="centered")

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Import necessary libraries
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Model options
Model_options = {
    "gemma2-9b-it": "Gemma 2 9B (Google)",
    "llama-3.3-70b-versatile": "Llama 3.3 70B Versatile (Meta)",
    "llama-3.1-8b-instant": "Llama 3.1 8B Instant (Meta)",
    "llama-guard-3-8b": "Llama Guard 3 8B (Meta)",
    "llama3-70b-8192": "Llama 3 70B 8192 (Meta)",
    "llama3-8b-8192": "Llama 3 8B 8192 (Meta)",
}

# PDF processing functions
def process_pdf(uploaded_file):
    # Save the uploaded file temporarily
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    # Load the saved PDF
    loader = PyPDFLoader(temp_path)
    documents = loader.load()
    
    # Remove temp file
    os.remove(temp_path)
    return documents

def split_document(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30)
    return splitter.split_documents(documents)

def create_vector_db(chunks):
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vdb = FAISS.from_documents(chunks, embedding_model)
    return vdb

def merge_vector_dbs(vdbs):
    if not vdbs:
        return None
    merged_db = vdbs[0]
    for vdb in vdbs[1:]:
        merged_db.merge_from(vdb)
    return merged_db

def get_answer(history, question, model_name, vdb=None):
    # Generate sanitized messages for the API call (only include role and content)
    messages = []
    for msg in history:
        # Only keep standard properties that Groq API accepts
        sanitized_msg = {
            "role": msg["role"],
            "content": msg["content"]
        }
        messages.append(sanitized_msg)
    
    # If we have documents to reference, get relevant context
    if vdb:
        retriever = vdb.as_retriever(search_kwargs={"k": 3})
        docs = retriever.get_relevant_documents(question)
        
        if docs:
            # Format context from documents
            context = "\n\n".join([f"Document: {i+1}\n{doc.page_content}" for i, doc in enumerate(docs)])
            
            # Find the system message
            system_msg_idx = None
            for i, msg in enumerate(messages):
                if msg["role"] == "system":
                    system_msg_idx = i
                    break
            
            # Add context to system message
            if system_msg_idx is not None:
                messages[system_msg_idx]["content"] += f"\n\nUse the following information to answer the question if relevant:\n{context}"
            else:
                # If no system message, add one with context
                messages.insert(0, {
                    "role": "system",
                    "content": f"You are an assistant. Use the following information to answer the question if relevant:\n{context}"
                })
            
            # Return the docs for citation
            source_docs = docs
        else:
            source_docs = []
    else:
        source_docs = []
    
    # Call the Groq API
    client = Groq(api_key=GROQ_API_KEY)
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model_name,
    )
    answer = chat_completion.choices[0].message.content
    
    return answer, source_docs

# Initialize session state variables
if 'history' not in st.session_state:
    st.session_state.history = [
        {
            'role': 'system',
            'content': 'You are Thabet, the best assistant. You will answer every question while sounding fun.'
        }
    ]

if 'vdb' not in st.session_state:
    st.session_state.vdb = None

if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []

# UI
st.title("AI Assistant")

# Sidebar for settings and document uploads
with st.sidebar:
    st.header("Settings & Documents")
    
    # Model selection
    selected_model = st.selectbox(
        "Select a model", 
        options=list(Model_options.keys()), 
        format_func=lambda x: Model_options[x]
    )
    
    # Document upload section
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF documents for context", 
        type="pdf", 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # Show list of uploaded files
        file_names = [file.name for file in uploaded_files]
        new_files = [f for f in file_names if f not in st.session_state.processed_files]
        
        if new_files and st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                all_chunks = []
                vdbs = []
                
                # Process each file
                for uploaded_file in uploaded_files:
                    if uploaded_file.name in new_files:
                        st.write(f"Processing {uploaded_file.name}...")
                        documents = process_pdf(uploaded_file)
                        chunks = split_document(documents)
                        all_chunks.extend(chunks)
                        
                        # Create a VDB for this file
                        file_vdb = create_vector_db(chunks)
                        vdbs.append(file_vdb)
                
                # If we already have a VDB, add the new one
                if st.session_state.vdb:
                    vdbs.append(st.session_state.vdb)
                
                # Merge all VDBs
                st.session_state.vdb = merge_vector_dbs(vdbs)
                
                # Update processed files list
                st.session_state.processed_files.extend(new_files)
                
                st.success(f"Processed {len(new_files)} new documents!")
    
    # Option to use documents
    use_docs = st.checkbox("Use document knowledge in responses", value=True)
    
    # Option to clear documents
    if st.session_state.vdb and st.button("Clear All Documents"):
        st.session_state.vdb = None
        st.session_state.processed_files = []
        st.success("Document knowledge cleared")

# Document context indicator
if st.session_state.processed_files:
    st.info(f"📚 Using knowledge from {len(st.session_state.processed_files)} documents")

# Chat display
for message in st.session_state.history:
    if message["role"] != "system":  # Don't display system messages
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # If this is a response with sources, show them
            if message.get("sources"):
                with st.expander("View Sources"):
                    for i, source in enumerate(message["sources"]):
                        st.markdown(f"**Source {i+1}**:\n{source}")

# Chat input
user_input = st.chat_input("Ask me anything...")

if user_input:
    # Add user message to chat
    user_message = {"role": "user", "content": user_input}
    st.session_state.history.append(user_message)
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate response
    with st.spinner("Thinking..."):
        # Use document knowledge if available and enabled
        vdb_to_use = st.session_state.vdb if use_docs else None
        
        answer, sources = get_answer(
            st.session_state.history, 
            user_input, 
            selected_model, 
            vdb=vdb_to_use
        )
    
    # Create assistant message with sources if available
    assistant_message = {
        "role": "assistant", 
        "content": answer
    }
    
    if sources:
        # Store source texts for display - don't include in message to API
        assistant_message["sources"] = [doc.page_content for doc in sources]
    
    # Add to history
    st.session_state.history.append(assistant_message)
    
    # Display assistant message
    with st.chat_message("assistant"):
        st.markdown(answer)
        
        # Display sources if available
        if sources:
            with st.expander("View Sources"):
                for i, source in enumerate(sources):
                    st.markdown(f"**Source {i+1}**:\n{source.page_content}")
    
    # Force a rerun to update the UI
    st.rerun()

# Footer
st.markdown("---")
st.caption("Powered by Groq LLMs • Built with Streamlit and LangChain")