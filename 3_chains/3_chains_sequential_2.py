# task 1: Ask a no of facts about an animal
# task 2: Translate the provided facts into a provided language
# All inputs (animal, count, language) come from user.

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# task 1: Ask a no of facts about an animal (dynamic: animal and count)
animal_facts_template = ChatPromptTemplate.from_messages([
    ("system", "You like Telling facts about {animal}. Make sure it should be short"),
    ("human", "Give {count} facts about {animal}"),
])

# task 2: Translate the provided facts into a provided language (dynamic: language)
translation_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can translate text into {language}"),
    ("human", "Translate the following text: {text}"),
])

# Keep user input (animal, count, language) in the state and add "text" = facts output.
# Then translation_template gets {text, language, ...} from that state.
chain = (
    RunnablePassthrough.assign(
        text=animal_facts_template | model | StrOutputParser(),
    )
    | translation_template
    | model
    | StrOutputParser()
)


def get_facts_and_translate(animal: str, count: int, language: str) -> str:
    """Takes animal, count, and language from user and returns translated facts."""
    return chain.invoke({"animal": animal, "count": count, "language": language})


# Option 1: Get user input from the terminal
if __name__ == "__main__":
    animal = input("Animal: ").strip() or "cat"
    count = input("Number of facts: ").strip() or "2"
    language = input("Translate to language: ").strip() or "Hindi"
    result = get_facts_and_translate(animal, int(count), language)
    print(result)