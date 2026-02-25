# task 1: Ask a no of facts about an animal
# task 2: Translate the provided facts into a provided language

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# task 1: Ask a no of facts about an animal (dynamic: animal and count)
animal_facts_template = ChatPromptTemplate.from_messages([
    ("system", "You like Telling facts about {animal}. Make sure it should be short"),
    ("human", "Give {count} facts about {animal}"),
])

# task 2: Translate the provided facts into a provided language
translation_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can translate text into {language}"),
    ("human", "Translate the following text: {text}"),
])

# Define additional processing steps using RunnableLambda
# count_words = RunnableLambda(lambda x: f"Word count: {len(x.split())}\n{x}")
prepare_for_translation = RunnableLambda(lambda output: {"text": output, "language": "hindi"})


# Create the combined chain using LangChain Expression Language (LCEL)
chain = animal_facts_template | model | StrOutputParser() | prepare_for_translation | translation_template | model | StrOutputParser() 

# Run the chain
result = chain.invoke({"animal": "cat", "count": 2})

# Output
print(result)