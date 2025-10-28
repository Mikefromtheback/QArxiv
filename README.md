# QArxiv

Десктопное приложение для QA по статьям с arxiv. <br>
Реализовано с использованием RAG, с возможностью гибридного поиска и применения реранкера. <br>
Реализовано без langchain и ollama, c использованием своих структур и FAISS, rank-bm25, sentence-transformers, llama-cpp-python. <br>
Реализовано без OCR, с использованием html представлений статей. <br>
С дополнительной информацией по архитектуре, анализом существующих решений, проведенными экспериментами, а также демо-видео можно ознакомиться по ссылке: <br>
https://docs.google.com/presentation/d/1PEIjSErqz8SbL-8Gcp-FhhzxkZT4zUTYlmhL3dUQA_U

---

## Структура репозитория
```
core/
  data/project_store.py           # локальное хранилище проектов
  htmlrag/                        # HTML → Markdown, чанкование
  ingestion/content_extractor.py  # скачивание HTML с ресурсами
  tools/paper_fetcher.py          # поиск на arXiv
  tools/paper_analyzer.py         # RAG пайплайн + вызов LLM
  llm_services.py                 # обертка над llama.cpp
config.py                         # env-настройки
main.py                           # точка входа (NiceGUI)
requirements.txt
state.py                          # связка бэкенда с UI
```

---

## Установка

### 0) Клонируйте репозиторий и зайдите в папку
```bash
git clone https://github.com/Mikefromtheback/QArxiv
cd <скачанная репа>
```

### 1) Создайте виртуальное окружение
- Linux/macOS:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
- Windows (PowerShell):
  ```powershell
  py -3 -m venv venv
  .\venv\Scripts\Activate.ps1
  ```

### 2) Установите зависимости из requirements.txt
```bash
pip install -r requirements.txt
```

### 3) Установите FAISS
- Если CPU:
  ```bash
  pip install faiss-cpu
  ```
- Если GPU:
  ```bash
  pip install faiss-gpu
  ```
  Примечания:
  - На Windows готовых GPU-колёс часто нет — используйте `faiss-cpu`.
  - На macOS, как правило, `faiss-cpu`.

### 4) Установите llama-cpp-python
Варианты зависят от ОС и того, хотите ли GPU.

- CPU (проще всего, все ОС):
  ```bash
  pip install --upgrade pip setuptools wheel
  pip install llama-cpp-python
  ```

- NVIDIA GPU (Linux, иногда Windows):
  1) Попробуйте готовые колёса (подберите вашу версию CUDA):
     ```bash
     # Пример:
     pip install llama-cpp-python-cu121   # для CUDA 12.1
     # или
     pip install llama-cpp-python-cu122   # для CUDA 12.2
     ```
     Если таких пакетов нет для вашей платформы, используйте сборку из исходников.
  2) Сборка из исходников с CUDA:
     - Требуются: CMake, компилятор, установленный CUDA Toolkit
     - Команда:
       ```bash
       CMAKE_ARGS="-DGGML_CUDA=on" pip install --no-cache-dir --force-reinstall llama-cpp-python
       ```

- Apple Silicon (macOS, Metal):
  - Обычно достаточно:
    ```bash
    pip install llama-cpp-python
    ```
  - Если нужно форсировать Metal:
    ```bash
    CMAKE_ARGS="-DGGML_METAL=on" pip install --no-cache-dir --force-reinstall llama-cpp-python
    ```

- Windows (CPU, самый простой путь):
  ```powershell
  pip install llama-cpp-python
  ```
  Для GPU на Windows — продвинутая тема: нужны CUDA + CMake + MSVC Build Tools, затем сборка как на Linux с `-DGGML_CUDA=on`.

Проверка:
```bash
python -c "from llama_cpp import Llama; print('llama.cpp OK')"
```

### 5) Особенности Linux (GTK/WebKit2 для нативного окна)
Для pywebview на Linux нужны системные библиотеки. Для Debian/Ubuntu:
```bash
sudo apt-get update
sudo apt-get install -y python3-gi python3-gi-cairo gir1.2-gtk-3.0 gir1.2-webkit2-4.0
```

Далее активируйте venv и добавьте `gi` из системного Python в venv (чтобы pywebview увидел GTK/WebKit):
```bash
# Создаем и активируем venv, если еще не:
python3 -m venv venv
source venv/bin/activate

# Используем системный Python для поиска gi и venv-Python для site-packages
ln -s $(/usr/bin/python3 -c "import gi; print(gi.__file__.replace('/__init__.py', ''))") \
      $(python -c "import site; print(site.getsitepackages()[0])")/gi
```

### 6) Модель GGUF для llama.cpp
Скачайте GGUF-модель и положите в `models/`. По умолчанию конфиг ожидает:
```
models/Qwen3-1.7B-Q8_0.gguf
```
Вы можете использовать любую совместимую GGUF-модель — путь укажите в .env.

---

## Настройка (.env)

Создайте файл `.env` в корне проекта (рядом с `main.py`). Пример:

```ini
# провайдер LLM
LLM_PROVIDER=llama.cpp

# llama.cpp (путь к GGUF и параметры генерации)
LLAMA_CPP_MODEL_PATH=models/Qwen3-1.7B-Q8_0.gguf
LLAMA_CPP_CTX=4096
LLAMA_CPP_THREADS=8
LLAMA_CPP_GPU_LAYERS=0        # >0 для частичной инференции на GPU (если сборка с CUDA/Metal)
LLAMA_CPP_CHAT_FORMAT=auto
LLAMA_CPP_TEMPERATURE=0.6
LLAMA_CPP_MAX_TOKENS=500
LLAMA_CPP_TOP_P=0.95
LLAMA_CPP_TOP_K=20

# RAG по умолчанию
RAG_EMBED_MODEL=Qwen/Qwen3-Embedding-0.6B
RAG_RERANK_MODEL=Qwen/Qwen3-Reranker-0.6B
RAG_CHUNK_SIZE_TOKENS=256
RAG_RETRIEVAL=hybrid            # emb | bm25 | hybrid
RAG_TOP_K=5
RAG_HYBRID_ALPHA=0.75
RAG_HYBRID_RRF_K=60
RAG_MAX_CONTEXT_TOKENS=1600
RAG_USE_RERANKER=false       # true — включает Qwen3-Reranker (тяжелее/медленнее)
RAG_RERANKER_TOP_N=25
```
---

## Запуск

```bash
python main.py
```


