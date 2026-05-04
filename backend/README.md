# Trading Agent Backend API

FastAPI-based backend for the Trading Agent platform providing stock analysis, news sentiment, and AI chat capabilities.

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- uv (recommended) or pip

### Installation

1. **Install dependencies with uv (recommended)**
   ```bash
   uv sync
   ```

   Or with pip:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

3. **Run the server**
   ```bash
   python -m app.main
   ```

   Or with uvicorn directly:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

4. **Access the API**
   - API: http://localhost:8000
   - Interactive docs: http://localhost:8000/docs
   - Alternative docs: http://localhost:8000/redoc

## 📁 Project Structure

```
backend/
├── app/
│   ├── config/          # Configuration and settings
│   ├── routes/          # API endpoints
│   ├── core/            # Business logic (analysis modules)
│   ├── models/          # Pydantic schemas
│   └── utils/           # Utility functions
├── .env                 # Environment variables (not in git)
├── .env.example         # Environment template
├── pyproject.toml       # Project dependencies (uv)
├── requirements.txt     # Pip dependencies
└── Dockerfile           # Container configuration
```

## 🔌 API Endpoints

### Analysis
- `POST /api/analysis/technical` - Technical analysis
- `POST /api/analysis/fundamental` - Fundamental analysis
- `POST /api/analysis/news` - News sentiment analysis

### Chat
- `POST /api/chat` - Chat with AI trading agent

### Health
- `GET /api/health` - Health check
- `GET /api/tickers` - Get popular tickers

## 🔧 Configuration

### Environment Variables

Required:
- `GEMINI_API_KEY` - Google Gemini API key

Optional:
- `NEWS_API_KEY` - News API key
- `LANGSMITH_API_KEY` - LangSmith API key
- `LANGSMITH_TRACING` - Enable tracing (true/false)
- `BACKEND_PORT` - Server port (default: 8000)
- `CORS_ORIGINS` - Allowed CORS origins
- `DEBUG` - Debug mode (true/false)
- `LOG_LEVEL` - Logging level

## 🐳 Docker Deployment

```bash
# Build image
docker build -t trading-agent-backend .

# Run container
docker run -p 8000:8000 --env-file .env trading-agent-backend
```

## 🧪 Development

### Code Style
```bash
black app/
flake8 app/
```

### Testing
```bash
pytest tests/
```

## 📚 API Documentation

### Documentation Files

- **[API_DOCUMENTATION.md](API_DOCUMENTATION.md)** - Complete API reference with all endpoints, parameters, and external APIs
- **[ROUTES_ARCHITECTURE.md](ROUTES_ARCHITECTURE.md)** - Detailed technical architecture, data flows, and internal workings
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Fast reference guide for common use cases
- **Interactive Docs** - http://localhost:8000/docs (when server is running)
- **ReDoc** - http://localhost:8000/redoc (alternative documentation)

### Quick Example

```bash
curl -X POST "http://localhost:8000/api/analysis/technical" \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "period": "1y"}'
```

### What Each Doc Covers

| Document | Best For |
|----------|----------|
| `API_DOCUMENTATION.md` | Understanding endpoints, parameters, responses, external APIs |
| `ROUTES_ARCHITECTURE.md` | Understanding code structure, data flow, internal architecture |
| `QUICK_REFERENCE.md` | Quick lookups, common examples, troubleshooting |
| `/docs` endpoint | Interactive testing, trying out endpoints |

## 🔐 Security

- API keys stored in environment variables
- CORS configured for specific origins
- Input validation via Pydantic models
- Non-root user in Docker container

## 📄 License

MIT License - See main project README
