import os
import json
import requests
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from telegram import Update
from telegram.ext import Application, ContextTypes, CommandHandler, MessageHandler, filters
from flask import Flask, request
from dotenv import load_dotenv

# === ПУТИ ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHUNK_PATH = os.path.join(DATA_DIR, "chunk.json")
INDEX_PATH = os.path.join(DATA_DIR, "index.faiss")

# Проверка файлов
if not os.path.exists(CHUNK_PATH) or not os.path.exists(INDEX_PATH):
    print("ОШИБКА: нет chunk.json или index.faiss")
    exit(1)

# === КОНФИГ ===
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    print("ОШИБКА: нет TELEGRAM_BOT_TOKEN в .env")
    exit(1)

API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    print("ОШИБКА: нет OPENROUTER_API_KEY в .env")
    exit(1)

URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

# === ЗАГРУЗКА ===
print("Загрузка модели...")
MODEL = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
index = faiss.read_index(INDEX_PATH)
with open(CHUNK_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)
print("Готов!")

# === ПАМЯТЬ ===
user_histories = {}

# === ПОИСК ===
def retrieve_chunks(query, k=3):
    query_embedding = MODEL.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]

# === LLM ===
def get_llm_response(query, context_chunks, history):
    context = "\n".join([c["text"] for c in context_chunks]) if context_chunks else "Нет данных"
    messages = [
        {"role": "system", "content": f"""
Ты — дружелюбный помощник.
Говори просто, по-человечески.
Поддерживай любой диалог.
Если есть контекст — используй его.
Контекст:
{context}
""".strip()}
    ]
    messages.extend(history[-10:])
    messages.append({"role": "user", "content": query})

    data = {
        "model": "deepseek/deepseek-chat",
        "messages": messages,
        "max_tokens": 400,
        "temperature": 0.8
    }

    try:
        response = requests.post(URL, headers=HEADERS, json=data, timeout=20)
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"].strip()
        history.append({"role": "assistant", "content": answer})
        return answer
    except Exception as e:
        print(f"LLM ошибка: {e}")
        return "Секунду, что-то с сетью... Напиши ещё раз!"

# === КОМАНДЫ ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Чем могу помочь?")

async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_histories.pop(chat_id, None)
    await update.message.reply_text("Ок, начинаем заново!")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    query = update.message.text.strip()

    await update.message.reply_chat_action("typing")

    relevant_chunks = retrieve_chunks(query, k=3)
    history = user_histories.get(chat_id, [])
    answer = get_llm_response(query, relevant_chunks, history)
    user_histories[chat_id] = history

    await update.message.reply_text(answer)

# === FLASK + WEBHOOK ===
app = Flask(__name__)
application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

# Добавляем хендлеры
application.add_handler(CommandHandler("start", start))
application.add_handler(CommandHandler("clear", clear))
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

# Webhook endpoint
@app.route(f"/{TELEGRAM_BOT_TOKEN}", methods=["POST"])
def webhook():
    update = Update.de_json(request.get_json(force=True), application.bot)
    application.process_update(update)
    return "OK", 200

@app.route("/")
def index():
    return "RAG-бот работает! Отправь /start в Telegram."

# === ЗАПУСК ===
if __name__ == "__main__":
    # Render даёт порт через $PORT
    port = int(os.environ.get("PORT", 5000))
    print(f"Запуск на порту {port}...")
    app.run(host="0.0.0.0", port=port)