from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.routes import router


app = FastAPI()

origins = [
    "https://github.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def main():
    return {"message": "Goodbye, world."}

app = FastAPI()
app.include_router(router)


if __name__ == "__main__":
    uvicorn.run(app=app, host="127.0.0.1", port=5555)
