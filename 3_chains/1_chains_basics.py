from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

prompt = ChatPromptTemplate.from_messages([
    ("system", "Tell me facts about {Animal}."),
    ("human", "Tell me {count} facts about {Animal}."),
])
# prompt = ChatPromptTemplate.from_messages(messages)
# template = prompt.invoke({"animal": "Dog", "count": 3})

# response = model.invoke(template)
chain = prompt | model | StrOutputParser()

result = chain.invoke({"Animal": "Elephant", "count": "1"})
print(result)   