from openai import OpenAI
import os

api_key = os.getenv("OPENAI_API_KEY")
print("API key found:", api_key is not None)

client = OpenAI(api_key=api_key)

resp = client.responses.create(
    model="gpt-5-mini",
    input="Say hello in five words."
)

print(resp.output_text)

