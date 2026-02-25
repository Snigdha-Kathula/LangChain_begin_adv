# I am practicing the parallel chains in langchain.

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnableSequence
from langchain_core.runnables.passthrough import RunnableAssign
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
summarize_chanel_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can summarize a youtube channel."),
    ("human", "Summarize the following youtube channel IN 2LINES: {channel_type}"),
])
creative_name_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can generate creative names for a youtube channel."),
    ("human", "Generate 2 creative name for a youtube channel about {channel_type}"),
])

creative_slogan_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can generate slogans for a business."),
    ("human", "Generate 2 slogan for a business about {business_type}"),
])

name_branch = (
    RunnableLambda(lambda x: creative_name_template.invoke(x))
    | model
    | StrOutputParser()
)

slogan_branch = (
    RunnableLambda(lambda x: creative_slogan_template.invoke(x))
    | model
    | StrOutputParser()
)
chain = (
    
    summarize_chanel_template | model | StrOutputParser()
    | RunnableParallel({"name": name_branch, "slogan": slogan_branch}) 
    | RunnableLambda(lambda x: f"Name: {x['name']}\nSlogan: {x['slogan']}")
)

result = chain.invoke({"channel_type": "technology"})
print(result)