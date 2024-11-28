from ultralytics import YOLO
import torch
from tqdm import tqdm

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# HIPERPARÁMETROS
epochs = 2
batch_size = 8
img_size = 640  
weights = "yolov8n.pt"  
data_yaml = "data/data.yaml"


model = YOLO(weights).to(device)

model.train(
    data=data_yaml,         
    epochs=epochs,          
    imgsz=img_size,         
    batch=batch_size,
    verbose = True
)