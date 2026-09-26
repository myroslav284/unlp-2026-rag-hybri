# 🇺🇦 Ukrainian NLP RAG System (UNLP 2026 Kaggle Challenge)

An offline, hardware-constrained Retrieval-Augmented Generation (RAG) pipeline built for the UNLP 2026 Shared Task on Document Understanding. 

This project demonstrates how to build a robust QA system that operates entirely offline, strictly within limited GPU constraints (2x T4), while maintaining high accuracy for Ukrainian language document processing.

## 🧠 System Architecture & Engineering Highlights

Unlike standard API-wrapper RAGs, this system is engineered for severe constraints:

*   **Custom Hybrid Search Engine:** Combines semantic understanding and exact keyword matching.
    *   *Dense Retrieval:* `SentenceTransformers` (60% weight) for semantic meaning.
    *   *Sparse Retrieval:* `TF-IDF` with n-grams (1,2) (40% weight) for precise terminology matching.
*   **Quantized LLM Inference:** Deployed **LapaLLM** via `llama.cpp` using GGUF formats to fit the model onto limited VRAM while keeping generation speeds high.
*   **Zero-Internet Dependency Management:** Engineered a custom wheel-loading script to install dependencies (`llama-cpp-python`, `scikit-learn`, etc.) dynamically in a fully offline Kaggle environment.
*   **Deterministic Generation:** Strict prompt engineering and temperature control (`temp=0.1`) paired with token constraints to guarantee clean Multiple-Choice (A-F) outputs.

## 🛠️ Tech Stack
*   **Language:** Python
*   **AI/LLM:** `llama.cpp` (GGUF), LapaLLM
*   **Embeddings & Search:** `SentenceTransformers`, `scikit-learn` (TF-IDF), `numpy`
*   **Data Processing:** `pandas`, `pypdf`

## 🚀 How it Works (Pipeline)
1. **Document Ingestion:** Parses PDFs and chunks text (40-word chunks with 10-word overlaps) for optimal context windows.
2. **Dual-Vectorization:** Pre-computes both dense embeddings (semantic) and TF-IDF matrices (lexical) for all document chunks.
3. **Hybrid Scoring:** Calculates cosine similarity for user queries across both vector spaces, merging scores via a weighted algorithm (`0.6 * semantic + 0.4 * lexical`).
4. **Context Injection:** Injects the highest-scoring chunk into a strictly formatted prompt.
5. **Inference:** LapaLLM processes the localized context and outputs the precise correct option.

## 📂 Repository Context
This code was executed natively in Kaggle's offline environment. The `inference_pipeline.py` script expects the specific directory structure provided by the UNLP competition dataset.