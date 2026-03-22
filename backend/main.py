from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.test import router as test_router

app = FastAPI(
    title="AuraAuth API",
    description="FastAPI backend for the AuraAuth face authentication system.",
    version="1.0.0",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Allow all origins in development. Tighten this in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(test_router)


# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "AuraAuth FastAPI backend. Visit /docs for the API explorer."}
