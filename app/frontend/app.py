import gradio as gr

mapeo_alergias = {
    'fried_egg': ['egg'],
    'omelette': ['egg'],
    'eggs_benedict': ['egg'],
    'lobster': ['shellfish'],
    'shrimp': ['shellfish'],
    'clam': ['shellfish'],
    'crab': ['shellfish'],
    'cheesecake': ['lactose'],
    'ice_cream': ['lactose'],
    'macaroni_and_cheese': ['lactose'],
    'peanut_butter': ['peanut'],
    'pad_thai': ['peanut'],
    'chicken_curry': [],
    'tiramisu': ['lactose', 'gluten'],
    'pizza': ['gluten', 'lactose'],
    'hamburger': ['gluten'],
    'sushi': ['fish'],
    'tempura': ['shellfish'],
    'baklava': ['nuts'],
    'greek_salad': ['lactose'],
    'paella': ['shellfish'],
}

def deteccion_alergias(yolo_detections, mapeo_alergias):
    resultados_alergias = {}
    for detection in yolo_detections:
        if detection in mapeo_alergias:
            resultados_alergias[detection] = mapeo_alergias[detection]
        else:
            resultados_alergias[detection] = []
    return resultados_alergias

# EJEMPLO
def predict_with_allergens(image):
    predictions = {
        "pizza": 0.85,
        "sushi": 0.10,
        "hamburger": 0.05,
    }
    detected_classes = [key for key, value in predictions.items() if value > 0.05]

    # POSIBLES ALÉRGENOS
    allergens = deteccion_alergias(detected_classes, mapeo_alergias)

    highlighted_text = "\n".join(
        f"• {category.capitalize()}: {confidence * 100:.1f}%" for category, confidence in predictions.items()
    )

    # FORMATEO PARA QUE SE VEA MEJOR
    allergens_detected = "\n".join(
        f"{k.capitalize()}: {', '.join(v)}" for k, v in allergens.items() if v
    ) or "Sin alérgenos detectados"

    predicted_category = max(predictions, key=predictions.get)

    return predicted_category, highlighted_text, allergens_detected

# INTERFAZ
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







