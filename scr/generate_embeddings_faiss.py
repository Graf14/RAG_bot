import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os  # Добавлен импорт os

# Папка и файл
DATA_DIR = "./"
CHUNK_FILE = "chunk.json"
INDEX_FILE = "index.faiss"

# Загрузка модели для эмбеддингов
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Чтение чанков из JSON
with open(os.path.join(DATA_DIR, CHUNK_FILE), "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Извлечение текстов чанков
texts = [chunk["text"] for chunk in chunks]

# Генерация эмбеддингов
embeddings = model.encode(texts, convert_to_numpy=True)

# Создание FAISS индекса
dimension = embeddings.shape[1]  # Размерность эмбеддингов
index = faiss.IndexFlatL2(dimension)  # Индекс для поиска по евклидовому расстоянию
index.add(embeddings)  # Добавление эмбеддингов в индекс

# Сохранение индекса
faiss.write_index(index, os.path.join(DATA_DIR, INDEX_FILE))

# Вывод информации
print(f"✅ Создано {len(chunks)} эмбеддингов и сохранено в {INDEX_FILE}.")