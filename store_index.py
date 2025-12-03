from dotenv import load_dotenv
import os
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, download_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

load_dotenv()

# Load correct keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAPI_API_KEY1")   # corrected

# Export keys to environment
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAPI_API_KEY1"] = OPENAI_API_KEY

# Load PDF data
extracted_data = load_pdf_files(data='data/')
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filter_data)

# Load embedding model
embeddings = download_embeddings()

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"

# Check/Create index
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,    # Make sure this matches your embedding model
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Connect to index
index = pc.Index(index_name)

# Upload embeddings into Pinecone
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name
)
