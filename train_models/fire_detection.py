import os
from ultralytics import YOLO

# ===============================
# Eliminar archivos de cache existentes en data_fire
# ===============================
cache_paths = [
    os.path.join("data_fire", "train", "labels.cache"),
    os.path.join("data_fire", "valid", "labels.cache")
]
for cache_file in cache_paths:
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print(f"Archivo de cache eliminado: {cache_file}")

# ===============================
# 1. Definir la ruta al archivo data.yaml
# ===============================
data_yaml_path = os.path.join("data_fire", "data.yaml")
data_yaml_path = os.path.abspath(data_yaml_path)
print("Usando archivo data.yaml en:", data_yaml_path)

# ===============================
# 2. Cargar el modelo preentrenado de YOLOv8
# ===============================
model = YOLO("yolov8n.pt")
print("Modelo YOLO cargado correctamente.")

# ===============================
# 3. Entrenar el modelo con ajustes optimizados para detección de incendios
# ===============================
model.train(
    data=data_yaml_path,        # Archivo de configuración de datos
    epochs=150,                  # Número de épocas
    imgsz=640,                  # Resolución de entrada
    batch=4,                    # Tamaño del batch
    project="fire_detection",   # Carpeta donde se guardarán los resultados
    name="yolov8_fire",         # Nombre del experimento
    device=0,                   # Dispositivo (usa 'cpu' si prefieres entrenar en CPU)
    half=True,                  # Entrena en fp16 para aprovechar la capacidad de la GPU
    lr0=1e-4,                   # Tasa de aprendizaje inicial reducida
    momentum=0.9,               # Momentum del optimizador
    weight_decay=0.0005,        # Regularización
    workers=8,                  # Número de workers para la carga de datos
    augment=True,               # Habilitar data augmentation
    verbose=True,               # Mostrar información detallada del entrenamiento
    save_period=10,             # Guardar un checkpoint cada 10 épocas
    patience=10                 # Early stopping: detiene el entrenamiento si no hay mejora en 5 épocas consecutivas
)
print("Entrenamiento completado.")

# ===============================
# 4. Evaluar el modelo (opcional)
# ===============================
metrics = model.val(data=data_yaml_path)
print("Evaluación completada. Resultados:")
print(metrics)

# ===============================
# 5. Exportar el modelo a ONNX para optimización en CPU
# ===============================
try:
    model.export(format="onnx", half=False, dynamic=False)
    print("Exportación a ONNX completada. El archivo exportado estará optimizado para inferencia en CPU.")
except Exception as e:
    print("Ocurrió un error durante la exportación del modelo:")
    print(e)