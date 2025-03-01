import os
import torch
from ultralytics import YOLO

# ===============================
# Verificación de recursos y preparación
# ===============================
print(f"GPU disponible: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No'}")
print(f"VRAM disponible: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f}GB" if torch.cuda.is_available() else "N/A")

# Limpieza de caché para evitar problemas de entrenamiento
cache_paths = [
    os.path.join("data_car", "train", "labels.cache"),
    os.path.join("data_car", "valid", "labels.cache")
]
for cache_file in cache_paths:
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print(f"Caché eliminado: {cache_file}")

# ===============================
# Configuración del entrenamiento
# ===============================
data_yaml_path = os.path.join("data_car", "data.yaml")
data_yaml_path = os.path.abspath(data_yaml_path)
print(f"Archivo de configuración: {data_yaml_path}")

model = YOLO("yolov8s.pt")
print("Modelo base cargado: YOLOv8s")

# ===============================
# Entrenamiento optimizado para accidentes
# ===============================
results = model.train(
    data=data_yaml_path,
    epochs=150,                # Más épocas para mejor convergencia
    imgsz=640,                 # Resolución estándar
    batch=16,                  # Batch size optimizado
    project="car_detection",   # Carpeta de resultados
    name="yolov8s_car_optimizado",  
    device=0,                  # GPU
    half=True,                 # Usar FP16 para optimizar memoria
    
    # Optimizaciones para evitar overfitting
    dropout=0.1,               # Dropout para regularización
    weight_decay=0.0005,       # Regularización de pesos
    
    # Hiperparámetros optimizados
    lr0=0.001,                 # Tasa de aprendizaje inicial más agresiva
    lrf=0.01,                  # Factor de la tasa de aprendizaje final (para scheduler)
    momentum=0.937,            # Momentum del optimizador
    
    # Augmentación de datos avanzada
    augment=True,              # Habilitar augmentación
    mosaic=1.0,                # Mosaico para diversidad
    mixup=0.1,                 # Mixup para robustez
    degrees=10.0,              # Rotación moderada
    translate=0.2,             # Traslación
    scale=0.2,                 # Escalado
    fliplr=0.5,                # Volteo horizontal
    perspective=0.0005,        # Perspectiva (menos para evitar distorsión)
    
    # Control de entrenamiento
    patience=20,               # Paciencia para early stopping (en épocas)
    save_period=25,            # Guardar modelo cada 25 épocas
    close_mosaic=10,           # Desactivar mosaico en las últimas épocas
    
    # Optimizadores
    cos_lr=True,               # Scheduler de tasa de aprendizaje cosenoidal
    
    # Evaluación
    val=True,                  # Validar durante entrenamiento
    
    # Recursos
    workers=8,                 # Trabajadores para carga de datos
    cache=True,                # Cachear imágenes para acelerar entrenamiento
    verbose=True               # Mostrar información detallada
)

print("Entrenamiento completado. Métricas finales:")
print(f"mAP50: {results.maps[0]:.4f}")
print(f"mAP50-95: {results.maps[1]:.4f}")

# ===============================
# Guardar el modelo en el formato y ubicación correctos
# ===============================
os.makedirs("models", exist_ok=True)
model.export(format="pytorch", imgsz=640)

# Copiar al directorio esperado por la aplicación
import shutil
best_pt = f"{model.trainer.save_dir}/weights/best.pt"
final_path = "models/model_car.pt"
shutil.copy(best_pt, final_path)
print(f"Modelo guardado en: {final_path}")

print("Proceso de entrenamiento completado con éxito.")