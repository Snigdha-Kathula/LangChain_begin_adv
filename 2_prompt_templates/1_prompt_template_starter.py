from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()

# Example 1: Promptwith Human Messages 
template = "Write a {tone} email to {company} expressing interest in the {position} position, mentioning {skill} as a key strength. Keep it to 4 lines max"
prompt = ChatPromptTemplate.from_template(template)
messages = prompt.invoke({
    "tone": "formal",
    "company": "Google",
    "position": "Software Engineer",
    "skill": "LangChain",
})

print(messages)

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
response = model.invoke(messages)
print(response.content)
