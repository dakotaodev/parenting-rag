import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.api.routers.query import router

app = FastAPI()
app.include_router(router)


@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    request.state.request_id = str(uuid.uuid4())
    response = await call_next(request)
    response.headers["X-Request-Id"] = request.state.request_id
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "request_id": request_id},
        headers={"X-Request-Id": request_id},
    )


@app.get("/")
async def root():
    return {"message": "Hello, World!"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "0.1.0"
    }


def main():
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
