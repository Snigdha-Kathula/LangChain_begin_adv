#conditional chains in langchain.

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# we will give a review of a product
# if the review is positive, respond with a thank you message
# if the review is negative, respond with a sorry message

review_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can respond to reviews."),
    ("human", "Review: {review} as positive, negative, neutral, escalate?. Just give the answer as positive or negative or neutral or escalate."),
])
positive_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can respond to positive reviews."),
    ("human", "Thank you for your positive review!"),
])
negative_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can respond to negative reviews."),
    ("human", "Sorry to hear that you had a negative experience. We will try to improve."),
])
neutral_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can respond to neutral reviews."),
    ("human", "Thank you for your review. We will try to improve."),
])
escalate_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can respond to escalate reviews."),
    ("human", "Sorry to hear that you had a negative experience. We will try to improve."),
])

# Each branch is (condition, runnable). The last argument is the default runnable when no condition matches.
branches = RunnableBranch(
    (lambda x: x.get("review") == "positive", positive_template | model | StrOutputParser()),
    (lambda x: x.get("review") == "negative", negative_template | model | StrOutputParser()),
    (lambda x: x.get("review") == "neutral", neutral_template | model | StrOutputParser()),
    (lambda x: x.get("review") == "escalate", escalate_template | model | StrOutputParser()),
    neutral_template | model | StrOutputParser(),  # default fallback
)

# Classify review -> string, wrap as {"review": classification}, then run the matching branch.
# No StrOutputParser() at the end; each branch already returns a string.
chain = (
    review_template
    | model
    | StrOutputParser()
    | RunnableLambda(lambda x: {"review": x.strip().lower()})
    | branches
)

result = chain.invoke({"review": "The product was great!"})
print(result)