import os
import json
import requests
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from dotenv import load_dotenv

# === ЗАГРУЗКА КЛЮЧЕЙ ===
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not TELEGRAM_BOT_TOKEN or not OPENROUTER_API_KEY:
    print("Ошибка: Проверь .env — TELEGRAM_BOT_TOKEN и OPENROUTER_API_KEY должны быть!")
    exit(1)

URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}

# === ПУТИ ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHUNK_PATH = os.path.join(DATA_DIR, "chunk.json")
INDEX_PATH = os.path.join(DATA_DIR, "index.faiss")

# Проверка файлов
if not os.path.exists(CHUNK_PATH):
    print(f"Ошибка: {CHUNK_PATH} не найден!")
    exit(1)
if not os.path.exists(INDEX_PATH):
    print(f"Ошибка: {INDEX_PATH} не найден!")
    exit(1)

# === ЗАГРУЗКА ===
print("Загрузка модели и данных...")
MODEL = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
index = faiss.read_index(INDEX_PATH)
with open(CHUNK_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)
print("Готов!")

# === ПАМЯТЬ ===
user_histories = {}

# === ПОИСК: ТОП-5 БЕЗ ФИЛЬТРОВ ===
def retrieve_chunks(query, k=10):
    query_embedding = MODEL.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    
    relevant = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(chunks):
            relevant.append({
                "text": chunks[idx]["text"],
                "doc_id": chunks[idx]["doc_id"],
                "page_num": chunks[idx]["page_num"],
                "distance": float(dist)
            })
    
    # Сортируем по релевантности (меньше расстояние = лучше)
    relevant.sort(key=lambda x: x["distance"])
    return relevant[:8]  # Топ-5

# === LLM: СВОБОДНЫЙ ДИАЛОГ + КОНТЕКСТ ===
def get_llm_response(query, context_chunks, history):
    # Формируем контекст с источниками
    if context_chunks:
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            src = f"{chunk['doc_id']}, стр. {chunk['page_num']}"
            context_parts.append(f"[{i}] {chunk['text']} (из {src})")
        context = "\n".join(context_parts)
    else:
        context = "Нет релевантной информации в базе."

    messages = [
 {"role": "system", "content": f"""
Ты — дружелюбный и умный помощник.
Говори просто, по-человечески.

ФОРМАТИРОВАНИЕ:
- Используй абзацы (пустые строки между ними).
- Делай нумерованные списки: 1. 2. 3.
- Делай отступы пробелами (4 пробела перед строкой).
- НЕ используй **, ##, -, *, _, или другие символы Markdown.
- НЕ выделяй жирным, курсивом, заголовками.
- Пиши как в обычном сообщении — чистый текст.

ПРАВИЛА:
1. Отвечай ТОЛЬКО по контексту.
2. Если пользователь спрашивает про технику — отвечай коротко и по делу.
3. Если нет — спроси: "Чем помочь?"
4. Не придумывай.
5. НЕ говори, что информация "из инструкции", "из базы", "в руководстве", ".Вообще не упонимай что данные берёшь из руководства
  

Контекст:
{context}
""".strip()}
    ]
    messages.extend(history[-10:])
    messages.append({"role": "user", "content": query})

    data = {
        "model": "deepseek/deepseek-chat",
        "messages": messages,
        "max_tokens": 500,
        "temperature": 0.7
    }

    try:
        response = requests.post(URL, headers=HEADERS, json=data, timeout=20)
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"].strip()
        history.append({"role": "assistant", "content": answer})
        return answer
    except:
        return "Секунду, что-то с сетью... Напиши ещё раз!"

# === КОМАНДЫ ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_histories.pop(chat_id, None)  # Полная очистка
    await update.message.reply_text("Привет! Чем могу помочь?")

async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_histories.pop(chat_id, None)
    await update.message.reply_text("Память очищена. Начнём заново!")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    query = update.message.text.strip()

    await update.message.reply_chat_action("typing")

    # Поиск топ-5
    relevant_chunks = retrieve_chunks(query, k=10)
    
    history = user_histories.get(chat_id, [])
    answer = get_llm_response(query, relevant_chunks, history)
    user_histories[chat_id] = history

    await update.message.reply_text(answer)

# === ЗАПУСК ===
if __name__ == "__main__":
    print("Бот запущен...")
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("clear", clear))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()