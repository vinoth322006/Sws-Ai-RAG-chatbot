<![CDATA[<div align="center">

# 📚 RAG Chatbot System

**A production-ready Retrieval-Augmented Generation chatbot that answers questions from your PDF documents.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-FF6F00?logo=google-chrome&logoColor=white)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

*Upload PDFs → Automatic chunking & embedding → Ask questions → Get grounded answers with citations*

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [LLM Provider Setup](#-llm-provider-setup)
- [Environment Variables](#-environment-variables)
- [API Documentation](#-api-documentation)
- [Architecture Decisions](#-architecture-decisions)
- [Performance Tuning](#-performance-tuning)
- [Troubleshooting](#-troubleshooting)
- [FAQ](#-faq)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔍 Overview

The RAG Chatbot System is a full-stack application that enables users to upload PDF documents and ask natural-language questions about their content. It uses **Retrieval-Augmented Generation (RAG)** to ground LLM responses strictly in the uploaded material, minimising hallucinations.

### Key Features

| Feature | Description |
|---|---|
| 📄 **PDF Upload & Processing** | Drag-and-drop PDF upload with automatic text extraction via PyMuPDF |
| 🔪 **Smart Chunking** | Sliding-window chunking (500 chars / 50 overlap) preserves context across boundaries |
| 🧠 **Semantic Embeddings** | BGE-small-en-v1.5 model — optimised for CPU, only 130 MB |
| 🗃️ **Vector Store** | ChromaDB for persistent, high-performance similarity search |
| 🤖 **Multi-LLM Support** | Gemini, OpenRouter, Fireworks AI, Grok, and local GGUF models |
| 🎯 **Grounded Answers** | Strict prompt engineering ensures answers come only from retrieved context |
| 🔌 **REST API** | Clean FastAPI endpoints with automatic OpenAPI documentation |
| 🖥️ **Web UI** | Built-in chat interface — no separate frontend needed |
| ⚡ **Low-Spec Friendly** | Designed to run on 4 GB RAM machines without a GPU |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          CLIENT (Browser)                           │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────────┐   │
│  │  Upload UI   │  │  Chat UI    │  │  Document Manager UI     │   │
│  └──────┬──────┘  └──────┬──────┘  └────────────┬─────────────┘   │
└─────────┼────────────────┼──────────────────────┼─────────────────┘
          │                │                      │
          ▼                ▼                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     FastAPI  (app/main.py)                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ POST /upload  │  │ POST /chat   │  │ GET /documents           │  │
│  └──────┬───────┘  └──────┬───────┘  └────────────┬─────────────┘  │
│         │                 │                       │                 │
│         ▼                 ▼                       ▼                 │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                   Service Layer                              │    │
│  │  ┌────────────┐  ┌────────────┐  ┌─────────────────────┐   │    │
│  │  │ PDF Parser  │  │ Chunker    │  │ Retrieval Engine    │   │    │
│  │  │ (PyMuPDF)   │  │ (sliding   │  │ (cosine + MMR)      │   │    │
│  │  │             │  │  window)   │  │                     │   │    │
│  │  └──────┬─────┘  └─────┬──────┘  └──────────┬──────────┘   │    │
│  │         │               │                    │               │    │
│  │         ▼               ▼                    ▼               │    │
│  │  ┌──────────────────────────────────────────────────────┐   │    │
│  │  │              ChromaDB  (Vector Store)                 │   │    │
│  │  │    BGE-small-en-v1.5 embeddings  ·  Cosine distance  │   │    │
│  │  └──────────────────────────────────────────────────────┘   │    │
│  │                                                              │    │
│  │  ┌──────────────────────────────────────────────────────┐   │    │
│  │  │              LLM Provider (selectable)                │   │    │
│  │  │  Gemini │ OpenRouter │ Fireworks │ Grok │ Local GGUF  │   │    │
│  │  └──────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
          │                                          │
          ▼                                          ▼
   ┌─────────────┐                          ┌──────────────┐
   │  /uploads    │                          │  /data        │
   │  (PDF files) │                          │  (ChromaDB)   │
   └─────────────┘                          └──────────────┘
```

### Request Flow

```
User Question
     │
     ▼
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│ Embed Query  │────▶│ Search       │────▶│ Rank & Filter│
│ (BGE-small)  │     │ ChromaDB     │     │ top_k=5      │
└─────────────┘     └──────────────┘     └──────┬───────┘
                                                 │
                                                 ▼
                                         ┌──────────────┐
                                         │ Build Prompt  │
                                         │ context +     │
                                         │ question      │
                                         └──────┬───────┘
                                                 │
                                                 ▼
                                         ┌──────────────┐
                                         │ LLM Generate  │
                                         │ (grounded     │
                                         │  answer)      │
                                         └──────┬───────┘
                                                 │
                                                 ▼
                                          Response + Sources
```

---

## 🚀 Quick Start

Get up and running in **5 steps**:

### Step 1 — Clone the Repository

```bash
git clone <your-repo-url> rag-system
cd rag-system
```

### Step 2 — Configure Environment

```bash
cp .env.example .env
# Edit .env and add at least one API key (e.g. GEMINI_API_KEY)
```

### Step 3 — Start the Server

**Linux / macOS:**
```bash
chmod +x start.sh
./start.sh
```

**Windows:**
```cmd
start.bat
```

The script automatically:
- ✅ Checks for Python 3.10+
- ✅ Creates a virtual environment
- ✅ Installs all dependencies
- ✅ Creates required directories
- ✅ Launches the server

### Step 4 — Upload a PDF

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@your-document.pdf"
```

Or open `http://localhost:8000` in your browser and use the drag-and-drop UI.

### Step 5 — Ask Questions

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the main topics covered in the document?"}'
```

🎉 **That's it!** The system will retrieve relevant chunks and generate a grounded answer.

---

## 🔧 LLM Provider Setup

The system supports **five LLM providers**. Configure your preferred provider in `config/settings.json` or via environment variables.

### Google Gemini (Recommended — Free Tier Available)

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create an API key
3. Set it in your `.env`:
   ```
   GEMINI_API_KEY=your-key-here
   ```
4. Set `"provider": "gemini"` in `config/settings.json`

**Available Models:**
| Model | Speed | Quality | Free Tier |
|---|---|---|---|
| `gemini-1.5-flash` | ⚡ Fast | Good | ✅ Yes |
| `gemini-1.5-pro` | Medium | Excellent | ✅ Limited |
| `gemini-2.0-flash` | ⚡ Fast | Excellent | ✅ Yes |

### OpenRouter (Access 100+ Models)

1. Sign up at [OpenRouter](https://openrouter.ai/)
2. Go to **Keys** → create a new key
3. Set it in your `.env`:
   ```
   OPENROUTER_API_KEY=your-key-here
   ```
4. Set `"provider": "openrouter"` in `config/settings.json`

**Recommended Free Models:**
| Model | Size | Notes |
|---|---|---|
| `mistralai/mistral-7b-instruct:free` | 7B | Great all-rounder |
| `meta-llama/llama-3.1-8b-instruct:free` | 8B | Latest Llama |
| `google/gemma-2-9b-it:free` | 9B | Google's open model |

### Fireworks AI (Fast Inference)

1. Sign up at [Fireworks AI](https://fireworks.ai/)
2. Go to **API Keys** → create a new key
3. Set it in your `.env`:
   ```
   FIREWORKS_API_KEY=your-key-here
   ```
4. Set `"provider": "fireworks"` in `config/settings.json`

**Available Models:**
| Model | Speed | Notes |
|---|---|---|
| `llama-v3p1-8b-instruct` | ⚡ Very Fast | Default, good balance |
| `llama-v3p1-70b-instruct` | Medium | Higher quality |
| `mixtral-8x7b-instruct` | Fast | MoE architecture |

### Grok (xAI)

1. Sign up at [xAI Console](https://console.x.ai/)
2. Create an API key
3. Set it in your `.env`:
   ```
   GROK_API_KEY=your-key-here
   ```
4. Set `"provider": "grok"` in `config/settings.json`

### Local GGUF Model (Fully Offline)

Run without any API key using a local GGUF model:

1. Download a GGUF model (e.g., from [Hugging Face](https://huggingface.co/)):
   ```bash
   # Example: download TinyLlama (small, fast, ~600 MB)
   mkdir -p models
   # Place your .gguf file in the models/ directory
   ```
2. Update `config/settings.json`:
   ```json
   {
     "provider": "local",
     "local": {
       "model_path": "models/your-model.gguf",
       "n_ctx": 4096,
       "n_threads": 6,
       "n_gpu_layers": 0,
       "temperature": 0.7
     }
   }
   ```

**Recommended GGUF Models for Low-Spec Hardware:**
| Model | RAM Needed | Quality |
|---|---|---|
| TinyLlama-1.1B-Chat (Q4_K_M) | ~1 GB | Basic |
| Phi-3-mini-4k (Q4_K_M) | ~2.5 GB | Good |
| Mistral-7B-Instruct (Q4_K_M) | ~4.5 GB | Excellent |

---

## 📋 Environment Variables

All environment variables and their defaults:

| Variable | Description | Default | Required |
|---|---|---|---|
| `GEMINI_API_KEY` | Google Gemini API key | — | If using Gemini |
| `OPENROUTER_API_KEY` | OpenRouter API key | — | If using OpenRouter |
| `FIREWORKS_API_KEY` | Fireworks AI API key | — | If using Fireworks |
| `GROK_API_KEY` | xAI Grok API key | — | If using Grok |
| `HOST` | Server bind address | `0.0.0.0` | No |
| `PORT` | Server port | `8000` | No |
| `LOG_LEVEL` | Logging level (`debug`, `info`, `warning`, `error`) | `info` | No |
| `LOCAL_MODEL_PATH` | Path to local GGUF model | `models/model.gguf` | If using local |
| `N_GPU_LAYERS` | GPU layers for llama.cpp (`0` = CPU only) | `0` | No |
| `N_THREADS` | CPU threads for local model | `6` | No |
| `N_CTX` | Context window size for local model | `4096` | No |

---

## 📡 API Documentation

The full interactive API documentation is available at `http://localhost:8000/docs` (Swagger UI) once the server is running.

### `POST /upload` — Upload a PDF Document

Upload a PDF file for processing. The system extracts text, chunks it, generates embeddings, and stores them in the vector database.

**Request:**
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@document.pdf"
```

**Response (200 OK):**
```json
{
  "status": "success",
  "filename": "document.pdf",
  "chunks": 42,
  "message": "Document processed and indexed successfully"
}
```

**Error Response (400):**
```json
{
  "detail": "Only PDF files are allowed"
}
```

**Error Response (422):**
```json
{
  "detail": "Could not extract text from the PDF. The file may be scanned/image-based."
}
```

---

### `POST /chat` — Ask a Question

Send a question and receive a grounded answer based on the uploaded documents.

**Request:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the refund policy?",
    "provider": "gemini"
  }'
```

**Request Body Schema:**
| Field | Type | Required | Description |
|---|---|---|---|
| `message` | string | ✅ | The question to ask |
| `provider` | string | ❌ | Override the default LLM provider |

**Response (200 OK):**
```json
{
  "reply": "According to the document, the refund policy states that...",
  "sources": [
    {
      "text": "Refund requests must be submitted within 30 days...",
      "source": "policy.pdf",
      "page": 3,
      "score": 0.87
    },
    {
      "text": "Full refunds are available for unused products...",
      "source": "policy.pdf",
      "page": 4,
      "score": 0.82
    }
  ]
}
```

**Error Response (400):**
```json
{
  "detail": "No documents have been uploaded yet. Please upload a PDF first."
}
```

---

### `GET /documents` — List Uploaded Documents

Retrieve a list of all processed documents.

**Request:**
```bash
curl http://localhost:8000/documents
```

**Response (200 OK):**
```json
{
  "documents": [
    {
      "filename": "policy.pdf",
      "chunks": 42,
      "uploaded_at": "2025-01-15T10:30:00Z",
      "size_bytes": 245760
    }
  ],
  "total": 1
}
```

---

### `DELETE /documents/{filename}` — Delete a Document

Remove a document and all its associated chunks from the vector store.

**Request:**
```bash
curl -X DELETE http://localhost:8000/documents/policy.pdf
```

**Response (200 OK):**
```json
{
  "status": "success",
  "message": "Document 'policy.pdf' and its 42 chunks have been deleted"
}
```

---

### `GET /settings` — Get Current Settings

Retrieve the current system configuration.

**Request:**
```bash
curl http://localhost:8000/settings
```

**Response (200 OK):**
```json
{
  "provider": "gemini",
  "chunk_size": 500,
  "chunk_overlap": 50,
  "retrieval": {
    "top_k": 5,
    "min_score": 0.25
  }
}
```

---

### `PUT /settings` — Update Settings

Update system configuration at runtime.

**Request:**
```bash
curl -X PUT http://localhost:8000/settings \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "openrouter",
    "retrieval": {
      "top_k": 10,
      "min_score": 0.3
    }
  }'
```

**Response (200 OK):**
```json
{
  "status": "success",
  "message": "Settings updated successfully"
}
```

---

### `GET /health` — Health Check

Check if the server is running and all components are operational.

**Request:**
```bash
curl http://localhost:8000/health
```

**Response (200 OK):**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "components": {
    "vector_store": "connected",
    "embedding_model": "loaded",
    "llm_provider": "gemini"
  }
}
```

---

## 🏛️ Architecture Decisions

### Chunking Strategy: Sliding Window (500 / 50)

**Why 500 characters?**
- Balances context richness with retrieval precision
- Smaller chunks (200–300) lose paragraph-level context
- Larger chunks (1000+) dilute relevance and waste LLM tokens
- 500 characters ≈ 80–100 words ≈ 1 meaningful paragraph

**Why 50-character overlap?**
- Prevents information loss at chunk boundaries
- If a key sentence spans two chunks, overlap ensures at least one chunk captures it fully
- 10% overlap is the sweet spot — higher overlap wastes storage with diminishing returns

```
Document Text:
|=====Chunk 1=====|
               |=====Chunk 2=====|
                              |=====Chunk 3=====|
         ↑ 50-char overlap ↑
```

### Embedding Model: BGE-small-en-v1.5

| Criteria | BGE-small | all-MiniLM-L6 | BGE-large |
|---|---|---|---|
| Size | 130 MB | 80 MB | 1.2 GB |
| Dimensions | 384 | 384 | 1024 |
| MTEB Score | 62.17 | 58.80 | 64.59 |
| CPU Speed | ~10ms/chunk | ~8ms/chunk | ~50ms/chunk |
| RAM Usage | ~300 MB | ~250 MB | ~2 GB |

**Decision:** BGE-small delivers near-SOTA quality at a fraction of the resource cost. It is 10× smaller than BGE-large while only 2.4 points behind on MTEB benchmarks. For a CPU-only RAG system on low-spec hardware, this is the optimal trade-off.

### Retrieval: Cosine Similarity + MMR

- **Cosine Similarity** measures semantic closeness between the query embedding and stored chunk embeddings
- **Maximal Marginal Relevance (MMR)** re-ranks results to reduce redundancy — so you don't get 5 near-identical chunks
- **`top_k=5`** returns enough context without overwhelming the LLM's context window
- **`min_score=0.25`** filters out chunks that are semantically unrelated (pure noise)

### Prompt Engineering: Strict Grounding

The system prompt enforces grounded responses:

```
You are a helpful assistant. Answer the user's question using ONLY
the context provided below. If the context does not contain enough
information to answer the question, say "I don't have enough
information in the uploaded documents to answer this question."

Do NOT use any prior knowledge. Do NOT make up information.
Always indicate which parts of the context support your answer.
```

This approach:
- ✅ Eliminates hallucinations from the LLM's training data
- ✅ Makes responses verifiable against source documents
- ✅ Provides clear "I don't know" signals when context is insufficient

### Vector Store: ChromaDB

**Why ChromaDB over FAISS / Pinecone / Weaviate?**
- **Zero configuration** — no external server, runs embedded
- **Persistent storage** — survives server restarts out of the box
- **Python-native** — no Docker, no JVM, pip install and go
- **Built-in embedding support** — integrates cleanly with sentence-transformers
- **Production-adequate** — handles up to ~1M documents efficiently

---

## ⚡ Performance Tuning

### For Low-Spec Hardware (4 GB RAM, no GPU)

```json
{
  "local": {
    "n_threads": 4,
    "n_ctx": 2048,
    "n_gpu_layers": 0
  },
  "retrieval": {
    "top_k": 3
  },
  "chunk_size": 400
}
```

**Tips:**
- Use `gemini` or `openrouter` providers — offloads LLM computation to the cloud
- The embedding model (BGE-small) uses ~300 MB RAM and runs on CPU
- Set `top_k: 3` to reduce context size sent to the LLM
- Reduce `chunk_size` to 400 for slightly faster embedding
- Close other memory-heavy applications while running

### For Mid-Spec Hardware (8 GB RAM, no GPU)

```json
{
  "local": {
    "n_threads": 6,
    "n_ctx": 4096,
    "n_gpu_layers": 0
  },
  "retrieval": {
    "top_k": 5
  },
  "chunk_size": 500
}
```

### For GPU-Equipped Hardware (NVIDIA)

```json
{
  "local": {
    "n_threads": 6,
    "n_ctx": 8192,
    "n_gpu_layers": 35
  },
  "retrieval": {
    "top_k": 8
  },
  "chunk_size": 800
}
```

**Tips:**
- Set `n_gpu_layers` to 35 for 7B models on 6 GB VRAM
- Increase `n_ctx` to 8192 for more context
- Larger `chunk_size` (800) works well with bigger context windows

### Optimising Upload Speed

For large PDFs (100+ pages):
- Processing is primarily CPU-bound (text extraction + embedding)
- Expect ~2–5 seconds per page on modern hardware
- A 100-page PDF produces ~200–400 chunks, taking 30–60 seconds to embed
- The system processes chunks in batches for efficiency

---

## 🔧 Troubleshooting

### 1. "Python 3.10 or higher is required"

**Cause:** System Python is older than 3.10.

**Solution:**
```bash
# Check your version
python3 --version

# Install Python 3.10+ via pyenv (Linux/macOS)
pyenv install 3.12.0
pyenv local 3.12.0

# Windows: download from https://www.python.org/downloads/
```

### 2. "ModuleNotFoundError: No module named 'app'"

**Cause:** Running uvicorn from the wrong directory.

**Solution:**
```bash
# Always run from the project root
cd rag-system
uvicorn app.main:app --reload

# Or use the start script which handles this automatically
./start.sh
```

### 3. "Could not extract text from the PDF"

**Cause:** The PDF is scanned/image-based (no embedded text layer).

**Solution:**
- Use a PDF with selectable text (not scanned images)
- Pre-process scanned PDFs with OCR (e.g., `ocrmypdf input.pdf output.pdf`)
- This is a known limitation — the system does not include OCR to keep dependencies minimal

### 4. "ChromaDB: Embedded database is already in use"

**Cause:** Another instance of the server is already running.

**Solution:**
```bash
# Find and kill the existing process
# Linux/macOS:
lsof -i :8000
kill <PID>

# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### 5. "GEMINI_API_KEY not set" or API key errors

**Cause:** Missing or invalid API key.

**Solution:**
1. Check that `.env` exists and contains the key
2. Verify the key is valid at the provider's dashboard
3. Ensure no extra spaces or quotes around the key in `.env`
4. Restart the server after changing `.env`

### 6. "torch: out of memory" during embedding

**Cause:** Not enough RAM for the embedding model.

**Solution:**
```bash
# Set environment variable to reduce memory usage
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Or reduce batch size in the code
# The default BGE-small model needs ~300 MB RAM
```

### 7. "Connection refused" when accessing http://localhost:8000

**Cause:** Server failed to start or is on a different port.

**Solution:**
1. Check the terminal for error messages
2. Ensure port 8000 is not in use: `netstat -tulnp | grep 8000`
3. Try a different port: `PORT=8080 ./start.sh`
4. Check firewall settings

### 8. Slow responses from local GGUF model

**Cause:** CPU inference is inherently slower than cloud APIs.

**Solution:**
- Increase `n_threads` to match your CPU core count
- Use a smaller model (TinyLlama 1.1B vs Mistral 7B)
- Use a quantized model (Q4_K_M is the best balance)
- Switch to a cloud provider for faster responses
- Enable GPU layers if you have a compatible GPU

### 9. "llama_cpp: model file not found"

**Cause:** The GGUF model file is not at the configured path.

**Solution:**
```bash
# Verify the file exists
ls -la models/

# Update the path in config/settings.json
# "model_path" must be relative to the project root or absolute
```

### 10. "Too many chunks" / Upload takes very long

**Cause:** Very large PDF with dense text.

**Solution:**
- Increase `chunk_size` to 800 or 1000 to reduce chunk count
- Consider splitting the PDF into smaller sections
- The embedding step is the bottleneck — larger chunk sizes = fewer embeddings = faster processing

### 11. "Port already in use" error on startup

**Cause:** Another service is using port 8000.

**Solution:**
```bash
# Use a different port
PORT=8080 ./start.sh

# Or on Windows
set PORT=8080
start.bat
```

### 12. Answers are not relevant or too vague

**Cause:** Retrieval is not returning good chunks.

**Solution:**
- Increase `top_k` from 5 to 8 or 10
- Lower `min_score` from 0.25 to 0.15 (more permissive)
- Decrease `chunk_size` for more precise matching
- Rephrase your question to use terms from the document
- Check if the document was processed correctly via `GET /documents`

---

## ❓ FAQ

### Q1: What file formats are supported?

**A:** Currently only **PDF** files are supported. The system uses PyMuPDF for text extraction, which handles most PDF formats including those with complex layouts, tables, and multi-column text. Scanned/image-only PDFs are not supported without prior OCR processing.

### Q2: Can I upload multiple PDFs?

**A:** Yes! Upload as many PDFs as you need. Each document's chunks are stored in the same vector database, and the system searches across all documents when answering questions. Source citations in responses indicate which document each piece of information comes from.

### Q3: Is my data sent to external servers?

**A:** It depends on your LLM provider:
- **Local GGUF model:** Everything stays on your machine — fully offline
- **Cloud providers (Gemini, OpenRouter, etc.):** Your question and relevant document chunks are sent to the LLM API for answer generation. The full documents are never uploaded to cloud providers.

### Q4: How much disk space does ChromaDB use?

**A:** Approximately **1.5 KB per chunk** for embeddings and metadata. A typical 50-page PDF produces ~100–200 chunks, using roughly 150–300 KB in the vector store. You can store thousands of documents without significant disk usage.

### Q5: Can I switch LLM providers without re-uploading documents?

**A:** Yes! Documents are embedded and stored independently of the LLM provider. You can switch providers at any time via the `/settings` endpoint or by editing `config/settings.json`. Only the answer-generation step uses the LLM — retrieval is handled entirely by the local embedding model and ChromaDB.

### Q6: What is the maximum PDF file size?

**A:** There is no hard-coded limit, but practical limits depend on your hardware:
- **4 GB RAM:** PDFs up to ~200 pages comfortably
- **8 GB RAM:** PDFs up to ~500 pages
- **16 GB RAM:** PDFs up to ~1000+ pages
- Processing time scales linearly with page count

### Q7: Can I run this on a Raspberry Pi or ARM device?

**A:** The FastAPI server, ChromaDB, and cloud LLM providers will work on ARM. However:
- `sentence-transformers` and PyTorch on ARM may require building from source
- Local GGUF models work on ARM via `llama-cpp-python` (compile with `CMAKE_ARGS="-DLLAMA_BLAS=ON"`)
- A Raspberry Pi 4 (4 GB) can run the system with a cloud LLM provider

### Q8: How do I reset everything and start fresh?

**A:** Delete the data directories and restart:
```bash
# Remove vector store and uploaded files
rm -rf data/ uploads/

# Restart the server — directories will be recreated automatically
./start.sh
```

### Q9: Can I use this behind a reverse proxy (Nginx, Caddy)?

**A:** Yes. Set `HOST=127.0.0.1` to bind only to localhost, then proxy from Nginx/Caddy:
```nginx
location / {
    proxy_pass http://127.0.0.1:8000;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
}
```

### Q10: Does the system support streaming responses?

**A:** The current version returns complete responses. Streaming support (Server-Sent Events) is planned for a future release and will enable token-by-token display in the chat UI.

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ using FastAPI, ChromaDB, and sentence-transformers**

[⬆ Back to Top](#-rag-chatbot-system)

</div>
]]>
