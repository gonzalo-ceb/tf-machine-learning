import gradio as gr
from ultralytics import YOLO
from PIL import Image

CATEGORY_MAP = {
    "macaron": "Contains: Egg, Nuts",
    "beignet": "Contains: Gluten, Lactose",
    "cruller": "Contains: Egg, Gluten, Lactose",
    "cockle_food": "Contains: Shellfish",
    "samosa": "Contains: Gluten, Possibly Nuts",
    "tiramisu": "Contains: Egg, Lactose",
    "tostada": "Contains: Gluten",
    "moussaka": "Contains: Lactose, Gluten",
    "dumpling": "Contains: Gluten, Possibly Egg",
    "sashimi": "Safe",
    "knish": "Contains: Gluten, Lactose",
    "croquette": "Contains: Gluten, Lactose",
    "couscous": "Contains: Gluten",
    "porridge": "Contains: Gluten, Possibly Lactose",
    "stuffed_cabbage": "Contains: Gluten, Possibly Lactose",
    "seaweed_salad": "Safe",
    "chow_mein": "Contains: Gluten, Possibly Egg",
    "rigatoni": "Contains: Gluten",
    "beef_tartare": "Safe",
    "cannoli": "Contains: Gluten, Lactose",
    "foie_gras": "Safe",
    "cupcake": "Contains: Gluten, Egg, Lactose",
    "osso_buco": "Safe",
    "pad_thai": "Contains: Egg, Nuts",
    "poutine": "Contains: Lactose, Gluten",
    "ramen": "Contains: Gluten, Possibly Egg",
    "pulled_pork_sandwich": "Contains: Gluten",
    "bibimbap": "Contains: Egg",
    "chicken_kiev": "Contains: Gluten, Lactose",
    "apple_pie": "Contains: Gluten, Possibly Lactose",
    "risotto": "Contains: Lactose",
    "fruitcake": "Contains: Gluten, Nuts",
    "chop_suey": "Safe",
    "haggis": "Contains: Gluten",
    "scrambled_eggs": "Contains: Egg",
    "frittata": "Contains: Egg, Lactose",
    "scampi": "Contains: Shellfish, Lactose",
    "sushi": "Safe",
    "orzo": "Contains: Gluten",
    "fritter": "Contains: Gluten, Lactose, Possibly Egg",
    "nacho": "Contains: Lactose, Possibly Gluten",
    "beef_stroganoff": "Contains: Lactose, Gluten",
    "beef_wellington": "Contains: Gluten, Lactose",
    "spring_roll": "Contains: Gluten",
    "savarin": "Contains: Gluten, Lactose",
    "crayfish_food": "Contains: Shellfish",
    "souffle": "Contains: Egg, Lactose",
    "adobo": "Safe",
    "streusel": "Contains: Gluten, Lactose",
    "deviled_egg": "Contains: Egg",
    "escargot": "Contains: Lactose",
    "club_sandwich": "Contains: Gluten",
    "carrot_cake": "Contains: Gluten, Nuts, Lactose",
    "falafel": "Safe",
    "farfalle": "Contains: Gluten",
    "terrine": "Safe",
    "poached_egg": "Contains: Egg",
    "gnocchi": "Contains: Gluten, Lactose",
    "bubble_and_squeak": "Contains: Lactose",
    "egg_roll": "Contains: Gluten, Egg",
    "caprese_salad": "Contains: Lactose",
    "sauerkraut": "Safe",
    "creme_brulee": "Contains: Lactose, Egg",
    "pavlova": "Contains: Egg, Lactose",
    "fondue": "Contains: Lactose",
    "scallop": "Contains: Shellfish",
    "jambalaya": "Safe",
    "tempura": "Contains: Gluten, Possibly Egg",
    "chocolate_cake": "Contains: Gluten, Lactose, Egg",
    "potpie": "Contains: Gluten, Lactose",
    "spaghetti_bolognese": "Contains: Gluten, Possibly Lactose",
    "sukiyaki": "Contains: Gluten",
    "applesauce": "Safe",
    "baklava": "Contains: Gluten, Nuts",
    "salisbury_steak": "Safe",
    "linguine": "Contains: Gluten",
    "edamame": "Safe",
    "coq_au_vin": "Contains: Lactose",
    "tamale": "Contains: Gluten",
    "macaroni_and_cheese": "Contains: Gluten, Lactose",
    "kedgeree": "Contains: Egg, Gluten",
    "garlic_bread": "Contains: Gluten, Lactose",
    "beet_salad": "Safe",
    "steak_tartare": "Safe",
    "vermicelli": "Contains: Gluten",
    "pate": "Safe",
    "pancake": "Contains: Gluten, Egg, Lactose",
    "tetrazzini": "Contains: Gluten, Lactose",
    "onion_rings": "Contains: Gluten",
    "red_velvet_cake": "Contains: Gluten, Lactose, Egg",
    "compote": "Safe",
    "lobster_food": "Contains: Shellfish",
    "chicken_curry": "Safe",
    "chicken_wing": "Safe",
    "caesar_salad": "Contains: Egg, Lactose",
    "succotash": "Safe",
    "hummus": "Safe",
    "fish_and_chips": "Contains: Gluten",
    "lasagna": "Contains: Gluten, Lactose",
    "lutefisk": "Safe",
    "sloppy_joe": "Contains: Gluten",
}



modelo = YOLO("data/best.pt")

def clasificar_imagen(imagen):
    """
    Classifies an image using the YOLO model and returns
    the predicted class along with its allergens.
    """
    try:
        resultados = modelo(imagen)
        prediccion = resultados[0].boxes.data.cpu().numpy()[0]  
        clases = resultados[0].names  

        clase_id = int(prediccion[5])
        probabilidad = round(float(prediccion[4]), 2)
        clase_nombre = clases[clase_id]

        allergens = CATEGORY_MAP.get(clase_nombre, "Allergens information not available")

        return (
            f"### Prediction: {clase_nombre}\n\n"
            f"- **Confidence**: {probabilidad * 100:.1f}%\n"
            f"- **Allergens**: {allergens}"
        )

    except IndexError:
        return "### No food detected in the image.\n\nPlease try again with another image."

    except Exception as e:
        return f"### An error occurred: {str(e)}"


interface = gr.Interface(
    fn=clasificar_imagen,
    inputs=gr.Image(type="pil", label="Upload a Food Image"),
    outputs=gr.Markdown(),
    title="🍽️ Food Classifier with YOLO",
    description=(
        "Identify food items in your images and get information about their allergens."
    ),
    theme="compact",
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)
