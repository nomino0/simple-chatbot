import os 
from groq import Groq

from dotenv import load_dotenv

load_dotenv()


GROQ_API_KEY = os.getenv('GROQ_API_KEY')

def llm_answer(history):
    client = Groq(api_key=GROQ_API_KEY)
    chat_completion = client.chat.completions.create(
        messages=history,
        model="llama-3.3-70b-versatile",
    )
    return chat_completion.choices[0].message.content  # Fixed typo here


def main():
    history = list()
    
    question = input('Posez Votre Question : ')
    
    while(question != '/bye'):
        user_prompt = {
             "role": "user",
            "content": question,
        }
        history.append(user_prompt)
        
        answer = llm_answer(history)
        print("AI : ", answer)
        ai_prompt = {
             "role": "assistant",
            "content": answer,
        }
        history.append(ai_prompt)
        question = input('\nPosez Votre Question : ')
    else:
        print("\nAI : ", "BYE BYE Don't come back please ")


if __name__ == "__main__":  # Fixed this line
    main()