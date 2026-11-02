from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")
text = """Artificial Intelligence is transforming healthcare, education, banking, agriculture and transportation by automating complex tasks."""
response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[{"role": "user", "content": f"Summarize this in one sentence:\n{text}"}]
)
print(response.choices[0].message.content)
