from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn

# Inicializar la aplicación FastAPI
app = FastAPI()

# Simulación de detecciones del modelo YOLO
def detect_food(image: bytes):
    # Simular detecciones y confianza
    detections = {
        "pizza": 0.85,
        "sushi": 0.10,
        "hamburger": 0.05,
    }
    return detections

# Ruta para recibir la imagen y retornar predicciones
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    predictions = detect_food(image_data)
    return JSONResponse(content={"predictions": predictions})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
