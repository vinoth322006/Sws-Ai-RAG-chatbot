
### Windows

```bat
start.bat
```

### Linux or macOS

```bash
chmod +x start.sh
./start.sh
```

The startup script creates a virtual environment, installs dependencies from `requirements.txt`, prepares required folders, and starts the server with:

```bash
uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload
```

Open the app at:

```text
http://127.0.0.1:8000
```

API docs are available at:

```text
http://127.0.0.1:8000/docs
```

## Configuration

Edit `config/settings.json` to choose the LLM provider and retrieval options. API keys can also be loaded from a `.env` file when present.

Common environment variables:

```env
GEMINI_API_KEY=your_key_here
OPENROUTER_API_KEY=your_key_here
FIREWORKS_API_KEY=your_key_here
GROK_API_KEY=your_key_here
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=info
```

For local GGUF inference, place a `.gguf` model in `models/` and point the local model settings to that file.

## API Endpoints

### Health Check

```bash
curl http://127.0.0.1:8000/api/health
```

### Upload PDFs

```bash
curl -X POST http://127.0.0.1:8000/api/upload \
  -F "files=@document.pdf"
```

The upload route accepts one or more files using the form field name `files`.

### List Documents

```bash
curl http://127.0.0.1:8000/api/documents
```

### Delete a Document

```bash
curl -X DELETE http://127.0.0.1:8000/api/documents/document.pdf
```

### Chat

```bash
curl -N -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"What is this document about?\",\"history\":[]}"
```

The chat endpoint returns a `text/event-stream` response. The first event includes retrieved sources, followed by streamed answer tokens.

### Read Settings

```bash
curl http://127.0.0.1:8000/api/settings
```

### Update Settings

```bash
curl -X POST http://127.0.0.1:8000/api/settings \
  -H "Content-Type: application/json" \
  -d "{\"provider\":\"gemini\"}"
```

## How It Works

1. PDFs are uploaded to the local data directory.
2. Text is extracted and split into overlapping chunks.
3. Chunks are embedded with a sentence-transformer model.
4. Embeddings and metadata are persisted in ChromaDB.
5. User questions are embedded and matched against stored chunks.
6. The best chunks are added to the prompt as context.
7. The selected LLM provider streams a grounded answer back to the UI.

## Troubleshooting

- If dependencies fail to install, confirm Python 3.10+ is on your `PATH`.
- If the server starts but answers are empty, upload at least one text-based PDF first.
- If a scanned PDF returns poor results, run OCR on the PDF before uploading it.
- If port `8000` is busy, set `PORT=8080` before running the startup script.
- If a cloud provider fails, verify the matching API key and selected provider in `config/settings.json`.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
