from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI()

# Suponiendo que tienes una función para procesar la imagen y hacer la predicción
def procesar_imagen(image):
    # Aquí iría tu código para procesar la imagen y hacer la predicción
    # Por ejemplo, cargar el modelo y predecir la categoría de la imagen
    prediccion = {
        "categoria": "pizza",
        "confianza": "85%",
        "alergenos": "gluten, lactosa"
    }
    return prediccion

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Procesar la imagen recibida
    image = await file.read()  
    prediccion = procesar_imagen(image)  

    return JSONResponse(content=prediccion)


