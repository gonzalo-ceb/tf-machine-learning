# Proyecto YOLO: Detección de Objetos y Clasificación de Imágenes

## Descripción General

Este proyecto aplica técnicas de Deep Learning para abordar problemas de **detección de objetos** y **clasificación de imágenes** utilizando **YOLOv11**. Incluye el proceso completo, desde la definición del problema hasta el despliegue en una aplicación sencilla basada en Gradio, con soporte de Docker para asegurar su portabilidad.

Nuestro repositorio: [TF Machine Learning](https://github.com/gonzalo-ceb/tf-machine-learning)

---

## Objetivo

Crear un sistema eficiente para la detección y clasificación de alimentos que pueda identificar alérgenos, facilitando su uso en entornos reales.

El proyecto combina visión por computadora y procesamiento del lenguaje natural (NLP) para un enfoque integral.

---

## Pasos Realizados

### 1. Conformación del Grupo
El equipo está compuesto por dos integrantes con diferentes entornos de trabajo (macOS y Windows). Solucionamos conflictos derivados del uso de `.venv` añadiendo estos directorios a `.gitignore`.

### 2. Definición del Problema
Queríamos:
1. Detectar alimentos en imágenes.
2. Identificar posibles alérgenos mediante una función.

### 3. Selección del Modelo
Usamos **YOLOv11** para visión por computadora y exploramos opciones de NLP con **Hugging Face** y **OpenAI**.

#### Consideraciones:
- **Fine-Tuning:** Entrenamos el modelo YOLOv11 con 84 epochs en nuestro dataset personalizado.
- **Labeling:** Preparamos el dataset con etiquetas en formato YOLO, usando herramientas como Roboflow.
- **Tamaño del Modelo:** Optamos por un modelo balanceado en precisión y rendimiento. 170.000 imágenes.
- **Entrenamiento:** Utilizamos CUDA en ordenador Windows con GPU 4090RTX de NVIDIA

### 4. Estructura del Dataset
El dataset fue obtenido de Kaggle: [iFood 2019 FGVC6 Dataset](https://www.kaggle.com/c/ifood-2019-fgvc6/data).

Pasos para preparar el dataset:
1. Descargar y descomprimir la carpeta del dataset de Kaggle.
2. Copiar los archivos dentro de la carpeta `data/` del repositorio.
3. Ejecutar los siguientes notebooks en orden:
   - `1_exploracion.ipynb`
   - `2_preparacion_yolo.ipynb`
