# AI Research Assistant
### RAG + Memory + Summarization + Classification

A production-grade full-stack AI system built with FastAPI,
Next.js, LangChain, HuggingFace, FAISS, and Groq.

---

## Quick Start (Local Development)

### 1. Clone and set up environment
```bash
git clone <your-repo-url>
cd ai-research-assistant

# Create root .env from template
cp .env.example .env

# Add your Groq API key to .env
# Get a free key at https://console.groq.com
```

### 2. Start the Backend
```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
# Edit .env and add GROQ_API_KEY

# Run the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend runs at: http://localhost:8000
Swagger docs at: http://localhost:8000/docs

### 3. Start the Frontend
```bash
cd frontend

# Install dependencies
npm install

# Environment is pre-configured for local dev
# frontend/.env.local already points to http://localhost:8000

# Start dev server
npm run dev
```

Frontend runs at: http://localhost:3000

---

## Docker Deployment (One Command)
```bash
# 1. Set your API key in .env
cp .env.example .env
# Edit .env: GROQ_API_KEY=your_key_here

# 2. Build and start everything
docker-compose up --build

# 3. Open http://localhost:3000
```

To stop:
```bash
docker-compose down
```

To stop and remove all data volumes:
```bash
docker-compose down -v
```

---

## API Endpoints Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET    | `/health` | Health check |
| POST   | `/api/v1/upload` | Upload and process a PDF |
| GET    | `/api/v1/documents` | List all documents |
| DELETE | `/api/v1/documents/{id}` | Delete a document |
| GET    | `/api/v1/vector-store/stats` | Vector index stats |
| POST   | `/api/v1/query` | Ask a question (RAG) |
| POST   | `/api/v1/query/stream` | Streaming answer (SSE) |
| GET    | `/api/v1/query/search` | Raw vector search |
| GET    | `/api/v1/query/history` | Get chat history |
| DELETE | `/api/v1/query/history` | Clear chat history |
| POST   | `/api/v1/analysis/summarize/{id}` | Summarize document |
| POST   | `/api/v1/analysis/classify/{id}` | Classify document |
| POST   | `/api/v1/analysis/analyze/{id}` | Summarize + classify |
| GET    | `/api/v1/analysis/categories` | List categories |

---

## Project Structure