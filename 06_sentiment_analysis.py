from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")
sentence = input("Enter a sentence: ")
response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[{"role": "user", "content": f"Identify the sentiment as positive, negative, or neutral: {sentence}"}]
)
print(response.choices[0].message.content)
