import gradio as gr

# Simulación de predicción con probabilidades
def predict_with_confidence(image):
    # Ejemplo de categorías y probabilidades
    predictions = {
        "Pizza 🍕": 0.85,
        "Sushi 🍣": 0.10,
        "Hamburguesa 🍔": 0.05,
    }

    # Crear barras visuales con texto destacado
    highlighted_text = []
    for category, confidence in predictions.items():
        # Si la confianza es mayor al 80%, verde; si no, rojo
        if confidence > 0.80:
            color = "green"
        else:
            color = "red"
        
        highlighted_text.append((f"{category}: {confidence*100:.1f}%", color))

    # Categoría principal predicha
    predicted_category = max(predictions, key=predictions.get)
    return predicted_category, highlighted_text

# Construcción de la interfaz en Gradio
with gr.Blocks(theme=gr.themes.Soft(primary_hue="teal")) as demo:
    gr.Markdown("# 🍲 **Food Classifier App**")
    gr.Markdown("Sube una imagen y descubre el alimento. ¡Es rápido y visual!")

    # Entrada de imagen
    with gr.Row():
        image_input = gr.Image(label="Sube tu imagen aquí", type="pil")

    # Salida: Resultado principal y barra visual
    with gr.Row():
        result = gr.Label(label="Categoría del alimento")  # Salida de la categoría principal
        confidence_chart = gr.HighlightedText(label="Confianza de Predicción")  # Barras visuales

    # Botón de predicción
    with gr.Row():
        submit_button = gr.Button("Clasificar 🚀")

    # Conexión del botón a la función de predicción
    submit_button.click(predict_with_confidence, inputs=image_input, outputs=[result, confidence_chart])

    # Pie de página
    gr.Markdown("---")
    gr.Markdown("**Creado por Gonzalo Celaya y Sandra González**")

# Lanza la app
demo.launch()





