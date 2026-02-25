from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSequence   
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

prompt = ChatPromptTemplate.from_messages([
    ("system", "Tell me facts about {Animal}."),
    ("human", "Tell me {count} facts about {Animal}."),
])

# --- Same chain using runnables with lambdas (see the inner steps) ---

# Step 1: Turn our input dict into a list of messages (what the prompt template does).
# Input: {"Animal": "Elephant", "count": "1"}  ->  Output: [SystemMessage(...), HumanMessage(...)]
format_prompt = RunnableLambda(lambda inputs: prompt.invoke(inputs))

# Step 2: Send those messages to the model and get a reply.
# Input: messages  ->  Output: AIMessage(content="...")
call_model = RunnableLambda(lambda messages: model.invoke(messages))

# Step 3: Take the model's message and return only the text (the content string).
# Input: AIMessage  ->  Output: "The capital of ..." (plain str)
parse_output = RunnableLambda(lambda msg: msg.content if hasattr(msg, "content") else str(msg))

# Pipe the runnables: output of one becomes input of the next (like prompt | model | parser).
# chain = format_prompt | call_model | parse_output
chain = RunnableSequence(format_prompt, call_model, parse_output)

result = chain.invoke({"Animal": "Elephant", "count": "1"})
print(result)   