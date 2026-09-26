"""
UNLP 2026 RAG Pipeline - Offline Hardware-Constrained Environment
Note: This script is optimized for the Kaggle execution environment with strict 
no-internet access and 2x T4 GPU limits. Dependencies are loaded locally via wheels.
"""

import os
import sys
import glob
import subprocess

print("Шукаємо драйвери...")
whl_files = glob.glob('/kaggle/input/**/*.whl', recursive=True)
if whl_files:
    safe_whls = [f for f in whl_files if 'numpy' not in f.lower()]
    wheels_dir = os.path.dirname(whl_files[0])
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install', '--no-index', '--find-links', wheels_dir] + safe_whls
    )
    print("✅ Драйвери успішно встановлено!")

# ==========================================
# 1. ІМПОРТ ТА ШЛЯХИ ДО ТЕСТОВИХ ДАНИХ
# ==========================================
import pandas as pd
import torch
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util
from llama_cpp import Llama

# Вимикаємо спам від PDF-читалки
import logging
logging.getLogger("pypdf").setLevel(logging.ERROR)

# Шляхи до даних змагання (TEST)
test_csvs = glob.glob('/kaggle/input/**/test.csv', recursive=True)
TEST_CSV_PATH = test_csvs[0]
INPUT_DIR = os.path.dirname(TEST_CSV_PATH)
TEST_PDF_DIR = os.path.join(INPUT_DIR, 'test')
SUBMISSION_PATH = '/kaggle/working/submission.csv'

# Автопошук моделей
gguf_files = glob.glob('/kaggle/input/**/*.gguf', recursive=True)
LLM_PATH = gguf_files[0] if gguf_files else ""
embedder_configs = glob.glob('/kaggle/input/**/modules.json', recursive=True)
EMBEDDER_PATH = os.path.dirname(embedder_configs[0]) if embedder_configs else ""

# ==========================================
# 2. ФУНКЦІЯ ЧИТАННЯ PDF (ЗВИЧАЙНІ 40 СЛІВ)
# ==========================================
def extract_and_chunk_pdfs(pdf_dir, chunk_size=40, overlap=10):
    pdf_files = glob.glob(os.path.join(pdf_dir, '**', '*.pdf'), recursive=True)
    chunks = []
    
    for pdf_path in pdf_files:
        doc_id = os.path.basename(pdf_path)
        try:
            reader = PdfReader(pdf_path)
            for page_index, page in enumerate(reader.pages):
                text = page.extract_text()
                if not text: continue
                
                words = text.strip().split()
                step = chunk_size - overlap
                for i in range(0, len(words), step):
                    chunks.append({
                        'Doc_ID': doc_id,
                        'Page_Num': page_index + 1,
                        'Chunk_Text': " ".join(words[i : i + chunk_size])
                    })
        except Exception:
            pass
    return chunks

# ==========================================
# 3. ЗАВАНТАЖЕННЯ, ВЕКТОРИЗАЦІЯ ТА TF-IDF
# ==========================================
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

print("Завантаження моделі Embeddings (Сенс)...")
embedder = SentenceTransformer(EMBEDDER_PATH)

print("Налаштування TF-IDF (Точні слова)...")
# Шукаємо окремі слова та пари слів
vectorizer = TfidfVectorizer(ngram_range=(1, 2)) 

print("Завантаження Lapa LLM на відеокарту...")
llm = Llama(model_path=LLM_PATH, n_gpu_layers=-1, n_ctx=2048, verbose=False)

print("Читання та розбиття тестових PDF...")
doc_chunks = extract_and_chunk_pdfs(TEST_PDF_DIR)

print(f"Створено {len(doc_chunks)} чанків. Будуємо Гібридний індекс...")
chunk_texts = [c['Chunk_Text'] for c in doc_chunks]

# 1. Векторизуємо нейромережею
chunk_embeddings = embedder.encode(chunk_texts, convert_to_tensor=True)
# 2. Векторизуємо класичним пошуком (TF-IDF)
tfidf_matrix = vectorizer.fit_transform(chunk_texts)

# ==========================================
# 4. ГІБРИДНИЙ ПОШУК ТА ГЕНЕРАЦІЯ
# ==========================================
test_df = pd.read_csv(TEST_CSV_PATH)
predictions = []

print(f"Починаємо обробку {len(test_df)} запитань гібридним методом...")
for index, row in test_df.iterrows():
    question_id = row['Question_ID']
    question_text = row['Question']
    
    options = {
        'A': str(row.get('A', '')), 'B': str(row.get('B', '')),
        'C': str(row.get('C', '')), 'D': str(row.get('D', '')),
        'E': str(row.get('E', '')), 'F': str(row.get('F', ''))
    }
    
    # 1. Оцінка нейромережі (пошук сенсу)
    q_emb = embedder.encode(question_text, convert_to_tensor=True)
    semantic_scores = util.cos_sim(q_emb, chunk_embeddings)[0].cpu().numpy()
    
    # 2. Оцінка TF-IDF (пошук точних слів)
    q_tfidf = vectorizer.transform([question_text])
    lexical_scores = sk_cosine(q_tfidf, tfidf_matrix)[0]
    
    # 3. ГІБРИДНЕ ОБ'ЄДНАННЯ (Беремо найкраще з обох світів)
    # Даємо 60% ваги нейромережі і 40% точним збігам слів
    hybrid_scores = (0.6 * semantic_scores) + (0.4 * lexical_scores)
    
    # Знаходимо абсолютного переможця
    best_idx = np.argmax(hybrid_scores)
    best_chunk = doc_chunks[best_idx]
    expanded_context = best_chunk['Chunk_Text']
    
    prompt = f"""Уважно прочитай текст і дай відповідь на запитання. Використовуй ТІЛЬКИ інформацію з тексту.

Текст:
{expanded_context}

Запитання: {question_text}

Варіанти відповідей:
A: {options['A']}
B: {options['B']}
C: {options['C']}
D: {options['D']}
E: {options['E']}
F: {options['F']}

Напиши тільки одну правильну літеру (A, B, C, D, E або F):"""

    messages = [
        {"role": "system", "content": "Ти помічник, який завжди відповідає строго однією літерою."},
        {"role": "user", "content": prompt}
    ]
    
    result = llm.create_chat_completion(messages=messages, max_tokens=3, temperature=0.1)
    predicted_letter = result['choices'][0]['message']['content'].strip()
    
    final_answer = 'A' 
    for char in predicted_letter:
        if char.upper() in ['A', 'B', 'C', 'D', 'E', 'F']:
            final_answer = char.upper()
            break
            
    predictions.append({
        'Question_ID': question_id,
        'Correct_Answer': final_answer,
        'Doc_ID': best_chunk['Doc_ID'],
        'Page_Num': best_chunk['Page_Num']
    })

# ==========================================
# 5. ЗБЕРЕЖЕННЯ САБМІТУ
# ==========================================
submission_df = pd.DataFrame(predictions)
submission_df.to_csv(SUBMISSION_PATH, index=False)
print(f"Готово! Файл {SUBMISSION_PATH} збережено.")