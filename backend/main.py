import random
from environment import Environment
import io
import pickle

import numpy as np
import PIL.Image
import PIL.ImageOps
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

_env = Environment()
with open(_env.models_paths_dict.get("mnist"), 'rb') as f:
    model = pickle.load(f)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "username": "black_champ",
        "name": "Black",
        "lastname": "Champ",
        "sex": "Every Day",
        "age": 0,
    }


@app.post("/predict-image/")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    pil_image = PIL.Image.open(io.BytesIO(contents)).convert('L')
    pil_image = PIL.ImageOps.invert(pil_image)
    pil_image = pil_image.resize((28, 28), PIL.Image.ANTIALIAS)
    img_array = np.array(pil_image).reshape(1, -1)
    prediction = model.predict(img_array)
    return {"prediction": int(prediction)}


@app.get("/items/{item_id}")
async def get_items_by_id(item_id: int | float):
    return {
        "item": item_id,
        "None": None
    }


@app.get("/items")
async def get_items():
    return {f"item_{i}": i for i in range(int(random.randint(10, 99)))}