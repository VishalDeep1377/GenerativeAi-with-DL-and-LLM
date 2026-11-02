from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")
messages = []
while True:
    user = input("You (type exit to stop): ")
    if user.lower() == "exit":
        break
    messages.append({"role": "user", "content": user})
    response = client.chat.completions.create(model="openai/gpt-oss-20b", messages=messages)
    reply = response.choices[0].message.content
    print("Bot:", reply)
    messages.append({"role": "assistant", "content": reply})
