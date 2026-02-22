from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

chat_history = []
system_message = SystemMessage("You are an helpful AI Assistant")
chat_history.append(system_message)

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(user_input))
    if user_input.lower() == "exit":
        break
    response = model.invoke(chat_history)
    chat_history.append(AIMessage(response.content))
    print(f"AI: {response.content}")