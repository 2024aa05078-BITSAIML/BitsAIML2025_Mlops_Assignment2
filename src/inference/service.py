from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import uvicorn

from src.inference.predict import predict_image

app = FastAPI(title="Cats vs Dogs Classifier API")


@app.get("/health")
def health_check():
    """
    Health check endpoint to verify the service is running.
    """
    return {"status": "healthy"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an image file and returns classification result.
    """

    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    result = predict_image(image)

    return result


if __name__ == "__main__":
    uvicorn.run("src.inference.service:app", host="0.0.0.0", port=8000, reload=True)
