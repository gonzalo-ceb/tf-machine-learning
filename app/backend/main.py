from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io

app = FastAPI()

# Función para simular el procesamiento de la imagen y hacer la predicción
def procesar_imagen(image):
    # Simulamos la predicción
    prediccion = {
        "categoria": "pizza",
        "confianza": "85%",
        "alergenos": "gluten, lactosa"
    }
    return prediccion

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Leemos los bytes del archivo de imagen
    image_bytes = await file.read()
    # Convertimos los bytes a una imagen que Pillow pueda manejar
    image = Image.open(io.BytesIO(image_bytes))

    # Simulamos el procesamiento y predicción
    prediccion = procesar_imagen(image)

    return JSONResponse(content=prediccion)
