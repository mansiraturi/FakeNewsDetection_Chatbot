import streamlit as st
from google.cloud import storage
from langchain.vectorstores.chroma import Chroma
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
import time

# Setup page configuration
st.set_page_config(page_title="Fake News Detection AI Chatbot", layout="centered")

# Show app startup status
startup_status = st.empty()
startup_status.info("Starting application...")

# GCS bucket details
BUCKET_NAME = "fakenewsrag"
GCS_PERSIST_PATH = "chroma/"
LOCAL_PERSIST_PATH = "./persistentdb/"

def ensure_gcp_authentication():
    """Ensure proper authentication for GCP services"""
    try:
        # Check if running in Cloud Run
        if os.environ.get('K_SERVICE'):
            startup_status.success("Running in Cloud Run with default service account")
            return True
            
        # Check if application default credentials are available
        client = storage.Client()
        # Test the credentials with a simple operation
        buckets = list(client.list_buckets(max_results=1))
        startup_status.success("GCP authentication successful")
        return True
    except Exception as e:
        startup_status.error(f"GCP Authentication failed: {str(e)}")
        st.error(f"Authentication Error: {str(e)}")
        st.info("Please make sure your service account has appropriate permissions")
        return False

# Verify authentication first
if not ensure_gcp_authentication():
    st.stop()

# Initialize GCS client
try:
    startup_status.info("Initializing GCS client...")
    storage_client = storage.Client()
except Exception as e:
    startup_status.error(f"Failed to initialize GCS client: {str(e)}")
    st.error(f"Storage Client Error: {str(e)}")
    st.stop()

def download_directory_from_gcs(gcs_directory, local_directory, bucket_name):
    """Download all files from a GCS directory to a local directory."""
    try:
        startup_status.info(f"Accessing bucket: {bucket_name}")
        bucket = storage_client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=gcs_directory))
        
        if not blobs:
            startup_status.warning(f"No files found in GCS path: {gcs_directory}")
            return False
        
        startup_status.info(f"Found {len(blobs)} files to download")
        for blob in blobs:
            if not blob.name.endswith("/"):  # Avoid directory blobs
                relative_path = os.path.relpath(blob.name, gcs_directory)
                local_file_path = os.path.join(local_directory, relative_path)
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                blob.download_to_filename(local_file_path)
                
        startup_status.success(f"Downloaded {len(blobs)} files from GCS")
        return True
    except Exception as e:
        startup_status.error(f"Error downloading from GCS: {str(e)}")
        st.error(f"Download Error: {str(e)}")
        return False

# Create local directory if it doesn't exist
os.makedirs(LOCAL_PERSIST_PATH, exist_ok=True)

# Check if we need to download from GCS
if not os.listdir(LOCAL_PERSIST_PATH):
    startup_status.info("Downloading vector database from Google Cloud Storage...")
    success = download_directory_from_gcs(GCS_PERSIST_PATH, LOCAL_PERSIST_PATH, BUCKET_NAME)
    if not success:
        startup_status.error("Failed to download vector database")
        st.error("Failed to download vector database. Please check your GCS configuration.")
        st.stop()
else:
    startup_status.info("Using existing local vector database")

# Step to use the data locally in retrieval
EMBEDDING_MODEL = "textembedding-gecko@003"
EMBEDDING_NUM_BATCH = 5

# Load embeddings and persisted data
try:
    startup_status.info("Initializing VertexAI embeddings...")
    embeddings = VertexAIEmbeddings(
        model_name=EMBEDDING_MODEL, batch_size=EMBEDDING_NUM_BATCH
    )
except Exception as e:
    startup_status.error(f"Failed to initialize embeddings: {str(e)}")
    st.error(f"Embedding Error: {str(e)}")
    st.stop()

# Load Chroma data from local persisted directory
try:
    startup_status.info("Loading Chroma database...")
    db = Chroma(persist_directory=LOCAL_PERSIST_PATH, embedding_function=embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
except Exception as e:
    startup_status.error(f"Failed to load vector database: {str(e)}")
    st.error(f"Database Error: {str(e)}")
    st.stop()

try:
    startup_status.info("Initializing conversation memory...")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')
    template = """
    You are a helpful AI assistant. Your task is to classify whether the news provided is real or fake based on the context provided 
    below. Then, explain why you made this classification.
    
    Context:
    {context}
    
    Question:
    {input}
    Answer:
    1. Classification: Please classify the news as either 'real' or 'fake' based on the context.
    2. Explanation: Provide a detailed explanation for why the news is classified as 'real' or 'fake'.
    """
    prompt = PromptTemplate.from_template(template)

    startup_status.info("Initializing Vertex AI LLM...")
    llm_gemini = VertexAI(
        model="gemini-1.5-pro",
        max_output_tokens=2048,
        temperature=0.2,
        top_p=0.8,
        top_k=40,
        verbose=True,
    )

    startup_status.info("Creating retrieval chain...")
    conversational_retrieval = ConversationalRetrievalChain.from_llm(
        llm=llm_gemini, retriever=retriever, memory=memory, verbose=False
    )
except Exception as e:
    startup_status.error(f"Failed to initialize conversation components: {str(e)}")
    st.error(f"LLM Setup Error: {str(e)}")
    st.stop()

# All initialization complete
startup_status.success("Application ready!")
time.sleep(1)  # Give users a moment to see the success message
startup_status.empty()  # Clear the status message

# Streamlit app UI
st.title("AI Assistant Chatbot")
st.markdown("""
### Fake News Classification
Ask the chatbot if a news article is real or fake. It will classify it and provide an explanation.
""")

# Initialize session state to store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages in a clean layout
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input box for user's query
user_input = st.chat_input("Enter your message here...")

if user_input:
    # Display user's message
    with st.chat_message("user"):
        st.markdown(f"**User**: {user_input}")

    # Store user's query in the chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Show thinking indicator with a more engaging visual
    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        thinking_placeholder.info("Processing your request... 🧠")

        try:
            # Get the AI assistant's response
            response = conversational_retrieval({"question": user_input})["answer"]

            # Replace thinking indicator with actual response
            thinking_placeholder.empty()
            st.markdown(f"**Assistant**: {response}")

            # Store AI's response in the chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception as e:
            thinking_placeholder.error(f"Error generating response: {str(e)}")
            st.session_state.messages.append({"role": "assistant", "content": f"Sorry, I encountered an error: {str(e)}"})

# Option to clear chat history with a styled button
st.markdown("""
---
""")
if st.button("Clear Chat History", key="clear_chat", use_container_width=True):
    st.session_state.messages = []
    try:
        memory.clear()
        st.success("Chat history cleared successfully! 🎉")
    except Exception as e:
        st.error(f"Error clearing memory: {str(e)}")
    st.experimental_rerun()
