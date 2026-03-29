from fastapi import FastAPI
from src.api.routers.query import router

app = FastAPI()
app.include_router(router)

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