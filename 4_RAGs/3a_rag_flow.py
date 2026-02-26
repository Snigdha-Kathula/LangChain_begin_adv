import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter


load_dotenv()
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
curr_dir = os.path.dirname(os.path.abspath(__file__))
persist_directory = os.path.join(curr_dir, "db", "dracula_db")
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

# load the file
loader = TextLoader(os.path.join(curr_dir, "documents", "Dracula.txt"))
documents = loader.load()


text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
chunks = text_splitter.split_documents(documents)
for i, chunk in enumerate(chunks):
    chunk.metadata["source"] = "Dracula.txt"
    chunk.metadata["doc_index"] = i+1

# vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_directory)
