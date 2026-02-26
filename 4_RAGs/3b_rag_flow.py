import os
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
curr_dir = os.path.dirname(os.path.abspath(__file__))
persist_directory = os.path.join(curr_dir, "db", "dracula_db")
vectorestore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

query = "What does dracula fear the most?"
retriever = vectorestore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
results =retriever.invoke(query)

#  Gemini Model to answer the question
gemini_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can answer questions about the text."),
    ("human", "Answer the question: {query}. Use the following text: {text} to answer the question. Keep the answer short and concise."),
])
template = prompt.invoke({"query": query, "text": results})
response = gemini_model.invoke(template)
print("RESPONSE :", response.content)

# for result in results:
#     print("RESULT :", result.metadata["doc_index"], result.metadata["source"], result.page_content)
