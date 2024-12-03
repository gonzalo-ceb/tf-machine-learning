import os
from ultralytics import YOLO
import torch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    data_yaml = "data/data.yaml"  
    weights = "yolov8n.pt"          

    # Hiperparámetros
    epochs = 300              
    batch_size = 64           
    img_size = 640             
    output_name = "train_results"  

    # Cargar el modelo YOLO
    print("Cargando modelo YOLO...")
    model = YOLO(weights).to(device)

    # Entrenamiento del modelo
    print("Iniciando entrenamiento...")
    model.train(
        data=data_yaml,         
        epochs=epochs,       
        imgsz=img_size,         
        batch=batch_size,       
        name=output_name,      
        device=device,          
        amp=True                
    )

    print(f"Entrenamiento completado. Resultados guardados en la carpeta 'runs/train/{output_name}/'")

if __name__ == '__main__':
    main()
