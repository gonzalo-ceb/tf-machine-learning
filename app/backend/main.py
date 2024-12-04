from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
from ultralytics import YOLO

app = FastAPI()

MODEL_PATH = "./data/yolo11n.pt" 
model = YOLO(MODEL_PATH)

def procesar_imagen(image):
    """
    Procesa la imagen con YOLO para hacer predicciones.
    """
    # Convertimos la imagen PIL a formato compatible con YOLO
    results = model(image)
    detections = results[0].boxes

    prediccion = {
        "categoria": [],
        "confianza": [],
        "alergenos": []
    }

    # Iteramos sobre las detecciones para recolectar los resultados
    for detection in detections:
        cls = int(detection.cls)  # Clase detectada
        confidence = float(detection.conf)  # Confianza
        class_name = model.names[cls]  # Nombre de la clase

        prediccion["categoria"].append(class_name)
        prediccion["confianza"].append(f"{confidence:.2f}")
        
        # Detectamos alérgenos si corresponde
        if class_name.lower() in ["peanut", "milk", "shellfish"]:  # Ejemplo de etiquetas para alérgenos
            prediccion["alergenos"].append(class_name)

    # Convertir listas a cadenas para el JSON
    prediccion["categoria"] = ", ".join(prediccion["categoria"])
    prediccion["confianza"] = ", ".join(prediccion["confianza"])
    prediccion["alergenos"] = ", ".join(prediccion["alergenos"]) if prediccion["alergenos"] else "No detectado"

    return prediccion

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint para manejar predicciones desde imágenes subidas.
    """
    # Leemos los bytes del archivo de imagen
    image_bytes = await file.read()
    # Convertimos los bytes a una imagen PIL
    image = Image.open(io.BytesIO(image_bytes))

    # Realizamos la predicción
    prediccion = procesar_imagen(image)

    return JSONResponse(content=prediccion)

