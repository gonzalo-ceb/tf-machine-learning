import gradio as gr
import torch 
from PIL import Image
from ultralytics import YOLO  

MODEL_PATH = "./data/yolo11n.pt"  
model = YOLO(MODEL_PATH)

def predict_with_yolo(image):
    """
    Realiza predicciones usando YOLO.
    """
    # Convertir la imagen a un formato compatible con YOLO
    results = model(image)
    detections = results[0].boxes  # Extrae las cajas de detección

    # Procesar resultados
    predicted_classes = []
    confidences = []
    allergens_detected = []  # Ajusta si tu modelo también detecta alérgenos

    for detection in detections:
        cls = int(detection.cls)  # Índice de la clase
        confidence = float(detection.conf)
        class_name = model.names[cls]  # Obtener el nombre de la clase

        predicted_classes.append(class_name)
        confidences.append(f"{confidence:.2f}")
        if class_name.lower() in ["peanut", "shellfish", "milk"]:  # Ejemplo de alérgenos
            allergens_detected.append(class_name)

    predicted_classes_str = ", ".join(predicted_classes)
    confidences_str = ", ".join(confidences)
    allergens_str = ", ".join(allergens_detected) if allergens_detected else "No detectado"

    return predicted_classes_str, confidences_str, allergens_str

# Crear la interfaz Gradio
with gr.Blocks() as demo:
    gr.Markdown("# **Food Allergy Detector App**")
    gr.Markdown("Sube una imagen para clasificar alimentos y verificar posibles alérgenos.")

    with gr.Row():
        image_input = gr.Image(label="Sube tu imagen aquí", type="pil")

    with gr.Row():
        result = gr.Label(label="Categoría del alimento")
        confidence_chart = gr.Textbox(label="Confianza de Predicción")
        allergens_output = gr.Textbox(label="Posibles alérgenos detectados")

    with gr.Row():
        submit_button = gr.Button("Clasificar y Verificar 🚀")

    submit_button.click(predict_with_yolo, inputs=image_input, outputs=[result, confidence_chart, allergens_output])

    gr.Markdown("---")
    gr.Markdown("**Creado por Gonzalo Celaya y Sandra González**")

demo.launch(server_name="0.0.0.0", server_port=7860)

