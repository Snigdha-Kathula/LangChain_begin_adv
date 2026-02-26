import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
curr_dir = os.path.dirname(os.path.abspath(__file__))
persist_directory = os.path.join(curr_dir, "db", "chroma_db_metadata")
vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
query = "What is sql?"
results = vectorstore.similarity_search(query)
print("content :", results[0].page_content)
print("metadata :", results[0].metadata)