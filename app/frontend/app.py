import gradio as gr
import requests

BACKEND_URL = "http://backend:8000/predict"

def predict_with_allergens(image):
    # Realizamos una solicitud POST al backend con la imagen
    response = requests.post(BACKEND_URL, files={"file": image})

    if response.status_code == 200:
        prediction = response.json()
        predicted_category = prediction["categoria"]
        allergens_detected = prediction["alergenos"]
        confidence = prediction["confianza"]
    else:
        predicted_category = "Error"
        allergens_detected = "Error"
        confidence = "Error"
    
    return predicted_category, confidence, allergens_detected

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

    submit_button.click(predict_with_allergens, inputs=image_input, outputs=[result, confidence_chart, allergens_output])

    gr.Markdown("---")
    gr.Markdown("**Creado por Gonzalo Celaya y Sandra González**")

demo.launch(server_name="0.0.0.0", server_port=7860)
