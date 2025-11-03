import requests
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import os

# Настройки API

import os
from dotenv import load_dotenv

load_dotenv()  # ← читает .env

API_KEY = os.getenv("OPENROUTER_API_KEY")
URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Модель для эмбеддингов
MODEL = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Файлы
DATA_DIR = "../data"  # ← на уровень выше, в data/
CHUNK_FILE = "chunk.json"
INDEX_FILE = "index.faiss"

# Загрузка индекса FAISS
index = faiss.read_index(os.path.join(DATA_DIR, INDEX_FILE))

# Чтение чанков из JSON
with open(os.path.join(DATA_DIR, CHUNK_FILE), "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Функция для поиска релевантных чанков
def retrieve_chunks(query, k=3):
    query_embedding = MODEL.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    retrieved_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]
    return retrieved_chunks

# Функция для вызова LLM с контекстом и историей
def get_llm_response(query, context_chunks, history):
    context = "\n".join([chunk["text"] for chunk in context_chunks])
    messages = [
        {"role": "system", "content": "Ты — дружелюбный ассистент, отвечай понятно и по существу. Используй следующий контекст для ответа: " + context}
    ]
    # Добавляем историю диалога
    messages.extend(history)
    # Добавляем текущий запрос
    messages.append({"role": "user", "content": query})
    
    data = {
        "model": "deepseek/deepseek-chat",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 300
    }
    
    try:
        response = requests.post(URL, headers=HEADERS, json=data, timeout=20)
        response.raise_for_status()
        result = response.json()
        # Добавляем ответ в историю
        history.append({"role": "assistant", "content": result["choices"][0]["message"]["content"]})
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"Ошибка при обращении к API: {e}"
    except KeyError:
        return f"Ошибка разбора ответа: {response.text}"

# Пример использования с диалогом
if __name__ == "__main__":
    history = []  # Список для хранения истории диалога
    while True:
        user_query = input("Задай вопрос (или 'выход' для завершения): ")
        if user_query.lower() == "выход":
            print("Диалог завершен.")
            break
        
        relevant_chunks = retrieve_chunks(user_query)
        print("Найденные чанки:")
        for chunk in relevant_chunks:
            print(f"- Файл: {chunk['doc_id']}, Страница: {chunk['page_num']}, Текст: {chunk['text']}")
        
        answer = get_llm_response(user_query, relevant_chunks, history)
        print("\nОтвет модели:")
        print(answer)