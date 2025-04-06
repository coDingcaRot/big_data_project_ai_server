from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import base64
from PIL import Image
from io import BytesIO
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model("./PickleFiles/sparse_early_stop_model.keras")
classes = ["apple", "banana", "grapes", "pear", "pineapple", "strawberry"]

class ImageData(BaseModel):
    image: str

def preprocess_image(image_b64):
    img_data = base64.b64decode(image_b64.split(",")[1])
    image = Image.open(BytesIO(img_data)).convert("L")
    image = Image.eval(image, lambda x: 255 - x)
    image = image.resize((256, 256))
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(1, 256, 256, 1)
    return image_array

@app.get("/")
def home():
    return {"message": "AI Drawing Prediction API is running!"}

@app.post("/predict")
def predict(data: ImageData):
    try:
        image = preprocess_image(data.image)
        prediction = model.predict(image)[0]  # Get the prediction array for the single image

        # Get the indices of the top 3 predictions and their confidence scores
        top_3_indices = np.argsort(prediction)[::-1][:3]
        top_3_predictions = [{"prediction": classes[i], "confidence": int(prediction[i] * 100)} for i in top_3_indices]

        return {"predictions": top_3_predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="localhost", port=5000)