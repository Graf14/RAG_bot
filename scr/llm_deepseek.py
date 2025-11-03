import requests


import os
from dotenv import load_dotenv

load_dotenv()  # ← читает .env

API_KEY = os.getenv("OPENROUTER_API_KEY")

url = "https://openrouter.ai/api/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}
quetion = input("Начнём диалог")
data = {
    "model": "deepseek/deepseek-chat",  # бесплатная модель на OpenRouter
    "messages": [
        {"role": "system", "content": "Ты — дружелюбный ассистент, отвечай понятно и по существу."},
        {"role": "user", "content": quetion}
    ],
    "temperature": 0.7,
    "max_tokens": 300
}

try:
    response = requests.post(url, headers=headers, json=data, timeout=20)
    response.raise_for_status()
    result = response.json()
    print("Ответ модели:")
    print(result["choices"][0]["message"]["content"])
except requests.exceptions.RequestException as e:
    print(f"Ошибка при обращении к API: {e}")
except KeyError:
    print(f"Ошибка разбора ответа: {response.text}")