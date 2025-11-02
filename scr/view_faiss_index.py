import json
import faiss
import numpy as np
import os

# Папка и файлы
DATA_DIR = "./"
CHUNK_FILE = "chunk.json"
INDEX_FILE = "index.faiss"

# Загрузка индекса FAISS
index = faiss.read_index(os.path.join(DATA_DIR, INDEX_FILE))

# Чтение чанков из JSON
with open(os.path.join(DATA_DIR, CHUNK_FILE), "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Проверка количества векторов
num_vectors = index.ntotal
print(f"Всего векторов в индексе: {num_vectors}")

# Вывод нескольких примеров (например, первые 5 чанков и их векторы)
for i in range(min(5, num_vectors)):
    vector = index.reconstruct(i)  # Восстановление вектора по индексу
    chunk = chunks[i]
    print(f"\n--- Чанк {chunk['chunk_id']} (Файл: {chunk['doc_id']}, Страница {chunk['page_num']}) ---")
    print(f"Текст: {chunk['text']}")
    print(f"Пример вектора (первые 5 значений): {vector[:5]}...")