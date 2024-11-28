import gradio as gr
import requests 

BACKEND_URL = "http://backend:8000/predict"  

def predict_with_allergens(image):
    response = requests.post(BACKEND_URL, files={"file": image})
    
    if response.status_code == 200:
        prediction = response.json()  
        predicted_category = prediction["category"]
        allergens_detected = prediction["allergens"]
        confidence_chart = prediction["confidence"]
    else:
        predicted_category = "Error"
        allergens_detected = "Error"
        confidence_chart = "Error"
    
    return predicted_category, confidence_chart, allergens_detected

# INTERFAZ GRADIO
with gr.Blocks(theme=gr.themes.Soft(primary_hue="teal")) as demo:
    gr.Markdown("# 🥗 **Food Allergy Detector App**")
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

demo.launch()
