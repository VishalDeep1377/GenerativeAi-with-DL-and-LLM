from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")
topic = input("Enter a topic: ")
response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[{"role": "user", "content": f"Explain {topic} in simple words suitable for beginners."}]
)
print(response.choices[0].message.content)
