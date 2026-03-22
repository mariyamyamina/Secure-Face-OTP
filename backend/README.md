# Backend — AuraAuth FastAPI Server

A Python FastAPI backend for the AuraAuth face authentication system.
This is a standalone Python project — completely independent from the Node.js monorepo.

## Tech stack

- Python 3.11+
- FastAPI
- Uvicorn (ASGI server)
- Pydantic v2 (data validation)

## Project structure

```
backend/
├── main.py             # FastAPI app + CORS + router registration
├── requirements.txt    # Python dependencies
├── routes/
│   ├── __init__.py
│   └── test.py         # GET /test — health/smoke test route
└── models/
    ├── __init__.py
    └── user.py         # Pydantic models for user data
```

## How to run

### 1. Create and activate a virtual environment

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the server

```bash
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.

## Available routes

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Root — confirms server is up |
| `GET` | `/test` | Smoke test endpoint |
| `GET` | `/docs` | Auto-generated Swagger UI |
| `GET` | `/redoc` | ReDoc API documentation |

## Example request

```bash
curl http://localhost:8000/test
# {"status":"ok","message":"AuraAuth FastAPI backend is running."}
```

## Adding new routes

1. Create a new file in `routes/`, e.g. `routes/auth.py`
2. Define an `APIRouter` and add your endpoints
3. Import and register the router in `main.py`:

```python
from routes.auth import router as auth_router
app.include_router(auth_router, prefix="/auth")
```

## Notes

- CORS is set to `allow_origins=["*"]` for development. Restrict this in production.
- The frontend is not connected to this backend yet — see `frontend/README.md` for
  details on the existing Express API that the React app currently uses.
