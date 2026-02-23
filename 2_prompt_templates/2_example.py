from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()

# Example 1: Prompt with Human Messages 
messages = [
    ("system", "You are a comedian who tells short jokes on topic {topic}."),
    ("human", "Tell me {Joke_count} jokes on topic {topic}."),
]
prompt = ChatPromptTemplate.from_messages(messages)
template = prompt.invoke({
    "topic": "AI",
    "Joke_count": 2,
})
print(template)

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
response = model.invoke(template)
print(response.content)