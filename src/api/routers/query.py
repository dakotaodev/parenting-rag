from fastapi import APIRouter

router = APIRouter(
    prefix="/api/v1",
    tags=["v1"]
)

@router.get("/query")
async def query():
    return {
        "message": "This is a query endpoint."
    }