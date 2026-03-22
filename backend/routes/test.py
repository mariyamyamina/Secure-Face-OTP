from fastapi import APIRouter

router = APIRouter()


@router.get("/test")
def test_endpoint():
    return {
        "status": "ok",
        "message": "AuraAuth FastAPI backend is running.",
    }
