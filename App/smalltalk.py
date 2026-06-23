from groq import Groq
import os
import re
from pathlib import Path
from dotenv import load_dotenv


env_path = Path(__file__).parent.parent / "Resource" / ".env"
load_dotenv(env_path )

GROQ_MODEL = os.getenv('GROQ_MODEL')

db_path = Path(__file__).parent.parent/"db.sqlite"

groq_client = Groq()

smalltalk_prompt = """You are a friendly, helpful assistant for an e-commerce chatbot platform.
When users engage in casual small talk (e.g. greetings, asking how you are, asking what you are,
whether you are a robot, what you do), respond in a warm, brief, conversational tone.
Keep responses short (1-2 sentences). Mention that you can help them with product searches
or answer questions about store policies if relevant.
Do not discuss anything unrelated to this role."""


def talk(query):
    completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": smalltalk_prompt
            },
            {
                "role": "user",
                "content": query
            }

        ],
        temperature=0.7,
        top_p=1
    )
    return completion.choices[0].message.content


if __name__ == "__main__":
    question= "are you a robot?"
    answer = talk(question)
    print(answer)