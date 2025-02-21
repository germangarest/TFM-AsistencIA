import cv2
import numpy as np
import torch
from ultralytics import YOLO

class UnifiedModel:
    def __init__(self, device="cpu"):
        self.device = device

        # Cargar modelos con la clase YOLO de Ultralytics
        self.model_car = YOLO("models/model_car.pt").to(self.device)
        self.model_fight = YOLO("models/model_fight.pt").to(self.device)
        self.model_fire = YOLO("models/model_fire.pt").to(self.device)

        # Configuración común
        self.classes = {
            0: ("Accidente", (255, 0, 0)),
            1: ("Pelea", (0, 255, 0)),
            2: ("Incendio", (0, 0, 255))
        }
        self.img_size = 640
        self.conf_threshold = 0.5

    def detect(self, img: np.ndarray):
        """
        Realiza la detección utilizando los tres modelos:
          - Realiza inferencia directa sobre la imagen.
          - Procesa los resultados filtrando detecciones por umbral de confianza.
          - Asigna manualmente el ID de clase a cada detección.
        Retorna un diccionario con las detecciones separadas por clase.
        """
        # Inicializar contenedor de detecciones para cada clase
        detections = {0: [], 1: [], 2: []}

        # Realizar inferencia en cada modelo
        results_car = self.model_car(img, verbose=False)[0]
        results_fight = self.model_fight(img, verbose=False)[0]
        results_fire = self.model_fire(img, verbose=False)[0]

        # Función interna para procesar los resultados de cada modelo
        def process_results(results, class_id):
            boxes = []
            # Iteramos sobre cada detección en el objeto de resultados
            for box in results.boxes:
                # box.conf es un tensor, se obtiene el valor escalar con .item()
                if box.conf.item() > self.conf_threshold:
                    # Se extraen las coordenadas en formato [x1, y1, x2, y2]
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    boxes.append([x1, y1, x2, y2, box.conf.item(), class_id])
            return boxes

        # Procesar detecciones para cada uno de los modelos
        detections[0] = process_results(results_car, 0)
        detections[1] = process_results(results_fight, 1)
        detections[2] = process_results(results_fire, 2)

        return detections