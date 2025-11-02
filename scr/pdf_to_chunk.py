import fitz  # PyMuPDF
import os
import re
import json

# Папки
DOCS_DIR = "docs/"
OUTPUT_DIR = "doc_process/"
DATA_DIR = "./"  # Корневая директория проекта для chunk.json

# Создание папок, если их нет
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Получение списка PDF-файлов
pdf_files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf")]

# Хранение всех чанков
all_chunks = []
chunk_id = 0

# Обработка каждого PDF
for pdf_file in pdf_files:
    pdf_path = os.path.join(DOCS_DIR, pdf_file)
    document = fitz.open(pdf_path)
    
    # Хранение текста
    pdf_text = {}

    # Обработка страниц
    for page_number in range(document.page_count):
        page = document.load_page(page_number)
        text = page.get_text("text")
        # Удаляем метки страниц перед очисткой
        text = re.sub(r'--- Страница \d+ ---', '', text, flags=re.MULTILINE)
        # Очистка текста
        clean_text = re.sub(r'[^a-zA-Zа-яА-Я0-9\s,\.-]', '', text)
        clean_text = re.sub(r'\.{2,}', '.', clean_text)
        clean_text = ' '.join(clean_text.split())
        pdf_text[page_number + 1] = clean_text

    document.close()
    
    # Чанкинг текста одной страницы (около 400 символов, до целого предложения)
    for page_number in pdf_text:
        page_text = pdf_text[page_number]
        if not page_text:
            continue
        
        sentences = re.split(r'(?<=[.!,\n])\s+(?=[A-ZА-Я])', page_text)
        current_chunk = ""
        for sentence in sentences:
            if not sentence.strip():
                continue
            sentence = sentence.strip() + "."
            temp_chunk = current_chunk + sentence + " "
            if len(temp_chunk) > 400 and current_chunk:
                all_chunks.append({
                    "doc_id": pdf_file,
                    "page_num": page_number,
                    "chunk_id": chunk_id,
                    "text": current_chunk.strip()
                })
                chunk_id += 1
                current_chunk = sentence + " "
            else:
                current_chunk = temp_chunk
        
        # Добавляем остаток, если он есть
        if current_chunk.strip() and len(current_chunk.strip()) >= 100:  # Минимальный порог
            all_chunks.append({
                "doc_id": pdf_file,
                "page_num": page_number,
                "chunk_id": chunk_id,
                "text": current_chunk.strip()
            })
            chunk_id += 1

# Сохранение всех чанков в один JSON файл
with open(os.path.join(DATA_DIR, "chunk.json"), "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)

# Вывод содержимого chunk.json для чтения (ограниченный до 10 чанков)
print("Первые 10 чанков из chunk.json:")
for i, chunk in enumerate(all_chunks[:10]):
    print(f"\n--- Чанк {chunk['chunk_id']} (Файл: {chunk['doc_id']}, Страница {chunk['page_num']}) ---\n{chunk['text']}\n")

print(f"\n✅ Обработка завершена! Чанки сохранены в chunk.json. Создано {len(all_chunks)} чанков.")