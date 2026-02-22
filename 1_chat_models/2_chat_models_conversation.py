from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

messages = [
    SystemMessage(content="You are an expert in Gen AI and you are helping the user with their questions."),
    HumanMessage(content="Give short answer to the question: What is LangChain?"),
]

response = model.invoke(messages)

print("Response: ", response)
print("Response content: ", response.content)

# Conversation with history
messages = [
    SystemMessage(content="You are an expert in Gen AI and you are helping the user with their questions."),
    HumanMessage(content="Give short answer to the question: What is LangChain?"),
    AIMessage(content="LangChain is an open-source framework designed to simplify the creation of applications powered by Large Language Models (LLMs) by connecting them with external data, tools, and agents."),
]