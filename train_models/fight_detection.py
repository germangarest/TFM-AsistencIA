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
    os.path.join("data_fight", "train", "labels.cache"),
    os.path.join("data_fight", "valid", "labels.cache")
]
for cache_file in cache_paths:
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print(f"Caché eliminado: {cache_file}")

# ===============================
# Configuración del entrenamiento
# ===============================
data_yaml_path = os.path.join("data_fight", "data.yaml")
data_yaml_path = os.path.abspath(data_yaml_path)
print(f"Archivo de configuración: {data_yaml_path}")

model = YOLO("yolov8s.pt")
print("Modelo base cargado: YOLOv8s")

# ===============================
# Entrenamiento optimizado para peleas
# ===============================
results = model.train(
    data=data_yaml_path,
    epochs=150,                   # Más épocas para mejor convergencia
    imgsz=640,                    # Resolución estándar
    batch=16,                     # Batch size para peleas (más complejo que accidentes)
    project="fight_detection",    # Carpeta de resultados
    name="yolov8s_fight_optimizado",  
    device=0,                     # GPU
    half=True,                    # Usar FP16 para optimizar memoria
    
    # Optimizaciones para evitar overfitting
    dropout=0.15,                 # Más dropout para peleas (más variabilidad)
    weight_decay=0.0007,          # Mayor regularización para evitar sobreajuste
    
    # Hiperparámetros optimizados
    lr0=0.0008,                   # Tasa de aprendizaje inicial ligeramente menor
    lrf=0.01,                     # Factor de la tasa de aprendizaje final
    momentum=0.937,               # Momentum del optimizador
    
    # Augmentación de datos avanzada (más agresiva para peleas)
    augment=True,                 # Habilitar augmentación
    mosaic=1.0,                   # Mosaico para diversidad
    mixup=0.15,                   # Más mixup para peleas (movimiento)
    degrees=15.0,                 # Mayor rotación para peleas
    translate=0.25,               # Mayor traslación
    scale=0.25,                   # Mayor escalado
    fliplr=0.5,                   # Volteo horizontal
    hsv_h=0.015,                  # Modificación de tono
    hsv_s=0.2,                    # Modificación de saturación
    hsv_v=0.2,                    # Modificación de brillo
    
    # Control de entrenamiento
    patience=20,                  # Paciencia para early stopping
    save_period=25,               # Guardar modelo cada 25 épocas
    close_mosaic=10,              # Desactivar mosaico en las últimas épocas
    
    # Optimizadores
    cos_lr=True,                  # Scheduler de tasa de aprendizaje cosenoidal
    
    # Pesos de pérdida específicos para peleas
    box=7.5,                      # Pérdida de caja
    cls=0.6,                      # Pérdida de clase
    dfl=1.5,                      # Distribution Focal Loss
    
    # Evaluación
    val=True,                     # Validar durante entrenamiento
    
    # Recursos
    workers=8,                    # Trabajadores para carga de datos
    cache=True,                   # Cachear imágenes para acelerar entrenamiento
    verbose=True                  # Mostrar información detallada
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
final_path = "models/model_fight.pt"
shutil.copy(best_pt, final_path)
print(f"Modelo guardado en: {final_path}")

print("Proceso de entrenamiento completado con éxito.")