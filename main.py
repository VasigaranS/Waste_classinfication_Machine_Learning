from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import cv2

import uvicorn

app= FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL=tf.keras.models.load_model("/Users/vasigarans/Desktop/waste_class/models/2")


class_names=['O', 'R']
@app.get("/ping")

async def ping():
    return "hello"


def read_file_as_image(data)->np.ndarray:
    image=np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict/")

async def predict(file: UploadFile=File(...)):
    image=read_file_as_image(await file.read())
    image =cv2.resize(image,(75,75))
    image_batch=np.expand_dims(image,0)
    print(image_batch)
    prediction=MODEL.predict(image_batch)
    predicted_class=class_names[np.argmax(prediction[0])]
    confidence=np.max(prediction[0])

    

    return {
        'class':predicted_class,
        'confidence':float(confidence)
    }



if __name__=="__main__":
    uvicorn.run(app,host='localhost',port=8000)


