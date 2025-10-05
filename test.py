# requirements:
# pip install python-dotenv langchain langchain-openai

from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1) load env (expects OPENAI_API_KEY in .env)
load_dotenv()

# 2) read local Markdown file with person's description
#    example file: person.md
#    ---
#    **Name:** Aisha  
#    **Age:** 29  
#    **Bio:** Urban planner who values equitable transit.  
#    **Traits:** direct, curious, data-driven  
#    ---
person_path = Path("john.md")
person_md = person_path.read_text(encoding="utf-8")

# 3) take a user input sentence as the topic prompt
topic_prompt = input("Topic prompt: ").strip()

# 4) build LangChain pipeline
system = (
    "You are an opinionated assistant. Use the provided person profile as context. "
    "Adopt the role requested by the user prompt. State a clear opinion, say whether you agree or disagree, "
    "and justify with brief reasoning. Avoid hedging."
)

template = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "user",
            "PERSON PROFILE (Markdown):\n{person_md}\n\n"
            "USER PROMPT:\n{topic_prompt}\n\n"
            "Respond in the first person as the role implied by the user prompt. "
            "Include: 1) stance (agree/disagree/nuanced), 2) 2–3 reasons, 3) a one-line takeaway."
        ),
    ]
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
chain = template | llm | StrOutputParser()

result = chain.invoke({"person_md": person_md, "topic_prompt": topic_prompt})

print("\n--- Opinion ---\n")
print(result)

