from openai import OpenAI
import os

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role":"user","content":"Hello, world!"}]
)
print(response.choices[0].message.content)