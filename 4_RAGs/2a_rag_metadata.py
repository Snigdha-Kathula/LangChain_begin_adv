# RAG basics: load document, split into chunks, create vector store
# File we read: 4_RAGs/documents/database.txt
# Install: pip install langchain-community langchain-text-splitters docarray

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()

# define a db folder and create a chroma_db folder inside it
curr_dir = os.path.dirname(os.path.abspath(__file__))
persist_directory = os.path.join(curr_dir, "db", "chroma_db_metadata")
file_path = os.path.join(curr_dir, "documents", "database.txt")
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

# Load the file
loader = TextLoader(file_path)
documents = loader.load()

# Add metadata using only simple types (str, int, float, bool) — Chroma does not allow dict/list in metadata
documents_with_metadata = []
for i, doc in enumerate(documents):
    doc.metadata["source"] = file_path  # string path
    doc.metadata["doc_index"] = i  # index of doc in the loaded list
    documents_with_metadata.append(doc)

# split the documents into chunks
splitter = CharacterTextSplitter(chunk_size=30, chunk_overlap=0)
chunks = splitter.split_documents(documents_with_metadata)
print(f"Split into {len(chunks)} chunks")

# create a vectore store 
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vectorstore = Chroma.from_documents(
    documents = chunks,
    embedding = embeddings,
    persist_directory = persist_directory
)

