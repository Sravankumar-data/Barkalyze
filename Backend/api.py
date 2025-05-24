from fastapi import FastAPI, File, Form, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import Annotated
from utils.common import process_and_store

app = FastAPI()

# CORS if you host this separately or on a different port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can tighten this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload/")
async def upload_image(
    background_tasks: BackgroundTasks,
    file: UploadFile,
    prediction: Annotated[str, Form()]
):
    image_bytes = await file.read()
    background_tasks.add_task(process_and_store, image_bytes, prediction.lower())
    return {"status": "processing"}
