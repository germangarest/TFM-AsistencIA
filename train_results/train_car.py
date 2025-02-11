import os
# Forzamos a TensorFlow a no usar ninguna GPU:
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import re
import numpy as np
import cv2  # Asegúrate de tener instalado opencv-python (pip install opencv-python)
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve, confusion_matrix

# Usaremos MobileNetV2, más ligero para CPU
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import (Dense, GlobalAveragePooling2D, Dropout, Input,
                                     TimeDistributed, LSTM)
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
                                        LearningRateScheduler)

# ==================== CALLBACK PARA TIEMPO RESTANTE ====================
import time

def format_time(seconds):
    """Formatea segundos a HH:MM:SS."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

class TimeRemainingCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.train_start_time = time.time()
        self.total_epochs = self.params.get('epochs', 1)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        self.steps = self.params.get('steps', None)
        self.batch_times = []
        self.last_batch_time = time.time()
        self.current_epoch = epoch

    def on_train_batch_end(self, batch, logs=None):
        current_time = time.time()
        batch_time = current_time - self.last_batch_time
        self.batch_times.append(batch_time)
        self.last_batch_time = current_time

        if self.steps and ((batch + 1) % 5 == 0 or batch == 0):
            avg_batch_time = np.mean(self.batch_times)
            batches_left = self.steps - (batch + 1)
            epoch_remaining = batches_left * avg_batch_time

            total_epochs_remaining = self.total_epochs - (self.current_epoch + 1)
            total_remaining = epoch_remaining + total_epochs_remaining * ((batch + 1) * avg_batch_time)

            print(f"[Época {self.current_epoch+1}] Batch {batch+1}/{self.steps} | "
                  f"Tiempo restante en la época: {format_time(epoch_remaining)} | "
                  f"Tiempo total restante estimado: {format_time(total_remaining)}")

    def on_epoch_end(self, epoch, logs=None):
        epoch_duration = time.time() - self.epoch_start_time
        print(f"Época {epoch+1} completada en {format_time(epoch_duration)}")

    def on_train_end(self, logs=None):
        total_duration = time.time() - self.train_start_time
        print(f"Entrenamiento completado en {format_time(total_duration)}")

# ==================== CONFIGURACIÓN ====================
class Config:
    # Parámetros del modelo y entrenamiento (ajustados para CPU)
    IMG_SIZE = 160              # Resolución: 160x160 (menor que 224 para acelerar)
    BATCH_SIZE = 2              # Tamaño del lote reducido
    EPOCHS = 20                 # Menor número de épocas
    INITIAL_LR = 2e-4           # Learning rate inicial
    MIN_LR = 1e-6               # Learning rate mínimo
    VALIDATION_SPLIT = 0.2      # Fracción para validación
    PRECISION_THRESHOLD = 0.6   # Umbral para clasificación
    USE_MIXED_PRECISION = False # Precisión mixta desactivada
    USE_CLASS_WEIGHTS = True    # Usar pesos de clases para balancear
    UNFREEZE_LAYERS = 0         # Para MobileNetV2, congelamos todo el modelo base

    # Parámetros de Data Augmentation
    ROTATION_RANGE = 10         
    SHIFT_RANGE = 0.1           
    ZOOM_RANGE = 0.15           
    BRIGHTNESS_RANGE = [0.85, 1.15]

    # Parámetros para secuencias de video
    SEQUENCE_LENGTH = 5         # Número de frames por secuencia reducido
    FRAME_STRIDE = 2            # Stride

    @classmethod
    def setup(cls):
        print("\nForzando uso de CPU (no se detectarán GPUs).")
        tf.config.set_soft_device_placement(True)
        try:
            tf.config.threading.set_inter_op_parallelism_threads(4)
            tf.config.threading.set_intra_op_parallelism_threads(4)
            print("Optimización de threads configurada para CPU.")
        except Exception as e:
            print(f"No se pudo configurar optimización de threads: {e}")

# ==================== FUNCIÓN PARA EXTRAER SECUENCIAS DE VIDEO ====================
def read_video_sequence(video_path, start_frame, sequence_length, target_size):
    """
    Lee una secuencia de frames desde un video a partir de 'start_frame'.  
    Si no se logran leer suficientes frames, se repite el último frame.  
    Se aplica el preprocesamiento propio de MobileNetV2.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el video: {video_path}")
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(sequence_length):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (target_size[0], target_size[1]))
        frame = preprocess_input(frame)
        frames.append(frame)
    cap.release()
    if len(frames) == 0:
        raise ValueError(f"No se pudo leer ningún frame del video: {video_path}")
    while len(frames) < sequence_length:
        frames.append(frames[-1])
    return np.array(frames)  # (SEQUENCE_LENGTH, height, width, 3)

def generate_video_sequences(video_dir, sequence_length, stride):
    """
    Genera secuencias de frames a partir de archivos de video en un directorio.  
    Cada secuencia se representa como una tupla: (ruta_del_video, frame_inicial)
    """
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    sequences = []
    video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir)
                   if f.lower().endswith(video_extensions)]
    video_files.sort()
    for video_file in video_files:
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"Advertencia: No se pudo abrir el video {video_file}. Se omite.")
            continue
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if total_frames < sequence_length:
            print(f"Advertencia: El video {video_file} tiene menos frames ({total_frames}) que SEQUENCE_LENGTH ({sequence_length}). Se omite.")
            continue
        for start in range(0, total_frames - sequence_length + 1, stride):
            sequences.append((video_file, start))
    return sequences

# ==================== GENERADOR PERSONALIZADO PARA SECUENCIAS DE VIDEO ====================
class VideoSequenceDataGenerator(tf.keras.utils.Sequence):
    """
    Generador de datos que carga secuencias de frames directamente desde videos.  
    Cada muestra es una secuencia extraída de un video usando una ventana deslizante.
    """
    def __init__(self, samples, labels, batch_size, target_size, sequence_length, augmenter=None, shuffle=True):
        self.samples = samples
        self.labels = labels
        self.batch_size = batch_size
        self.target_size = target_size  # (ancho, alto)
        self.sequence_length = sequence_length
        self.augmenter = augmenter
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def on_epoch_end(self):
        if self.shuffle:
            indices = np.arange(len(self.samples))
            np.random.shuffle(indices)
            self.samples = [self.samples[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]
    
    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))
    
    def __getitem__(self, index):
        batch_samples = self.samples[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]
        X = []
        for (video_path, start_frame) in batch_samples:
            frames = read_video_sequence(video_path, start_frame, self.sequence_length, self.target_size)
            if self.augmenter is not None:
                seed = np.random.randint(0, 100000)
                augmented_frames = []
                for frame in frames:
                    augmented_frame = self.augmenter.random_transform(frame, seed=seed)
                    augmented_frames.append(augmented_frame)
                frames = np.array(augmented_frames)
            X.append(frames)
        X = np.array(X)  # (batch_size, SEQUENCE_LENGTH, height, width, 3)
        y = np.array(batch_labels)
        return X, y

# ==================== CREACIÓN DE GENERADORES DE DATOS ====================
def create_video_sequence_generators():
    """
    Crea generadores de datos para entrenamiento y validación a partir de videos.  
    Se asume que el dataset está organizado en dos carpetas dentro de 'data_car':
      - 'Normal_Videos_for_Event_Recognition'
      - 'CrashAccidents'
    """
    data_dir = 'data_car'
    classes = ['Normal_Videos_for_Event_Recognition', 'CrashAccidents']
    train_samples = []
    train_labels = []
    val_samples = []
    val_labels = []
    stride = Config.FRAME_STRIDE
    
    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(data_dir, class_name)
        sequences = generate_video_sequences(class_path, Config.SEQUENCE_LENGTH, stride)
        np.random.shuffle(sequences)
        split_index = int(len(sequences) * (1 - Config.VALIDATION_SPLIT))
        train_sequences = sequences[:split_index]
        val_sequences = sequences[split_index:]
        train_samples.extend(train_sequences)
        train_labels.extend([class_idx] * len(train_sequences))
        val_samples.extend(val_sequences)
        val_labels.extend([class_idx] * len(val_sequences))
    
    # Data augmentation para entrenamiento
    train_augmenter = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=Config.ROTATION_RANGE,
        width_shift_range=Config.SHIFT_RANGE,
        height_shift_range=Config.SHIFT_RANGE,
        zoom_range=Config.ZOOM_RANGE,
        brightness_range=Config.BRIGHTNESS_RANGE,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_augmenter = None
    
    train_generator = VideoSequenceDataGenerator(
        samples=train_samples,
        labels=train_labels,
        batch_size=Config.BATCH_SIZE,
        target_size=(Config.IMG_SIZE, Config.IMG_SIZE),
        sequence_length=Config.SEQUENCE_LENGTH,
        augmenter=train_augmenter,
        shuffle=True
    )
    
    validation_generator = VideoSequenceDataGenerator(
        samples=val_samples,
        labels=val_labels,
        batch_size=Config.BATCH_SIZE,
        target_size=(Config.IMG_SIZE, Config.IMG_SIZE),
        sequence_length=Config.SEQUENCE_LENGTH,
        augmenter=val_augmenter,
        shuffle=False
    )
    
    if Config.USE_CLASS_WEIGHTS:
        total = len(train_labels)
        class_counts = np.bincount(train_labels)
        class_weights = {
            0: 1.0,
            1: (total / class_counts[1]) if class_counts[1] > 0 else 1.0
        }
        print(f"Pesos de clases calculados: {class_weights}")
    else:
        class_weights = None
    
    return train_generator, validation_generator, class_weights

def create_evaluation_sequence_generator(data_dir='data_car'):
    """
    Crea un generador para evaluación a partir de videos (sin data augmentation).  
    Se asume que el dataset en 'data_car' contiene las carpetas:
      - 'Normal_Videos_for_Event_Recognition'
      - 'CrashAccidents'
    """
    classes = ['Normal_Videos_for_Event_Recognition', 'CrashAccidents']
    samples = []
    labels = []
    stride = Config.FRAME_STRIDE
    
    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(data_dir, class_name)
        sequences = generate_video_sequences(class_path, Config.SEQUENCE_LENGTH, stride)
        for seq in sequences:
            samples.append(seq)
            labels.append(class_idx)
    
    eval_generator = VideoSequenceDataGenerator(
        samples=samples,
        labels=labels,
        batch_size=Config.BATCH_SIZE,
        target_size=(Config.IMG_SIZE, Config.IMG_SIZE),
        sequence_length=Config.SEQUENCE_LENGTH,
        augmenter=None,
        shuffle=False
    )
    return eval_generator

# ==================== DEFINICIÓN DEL MODELO LIVIANO ====================
def create_model(training=True):
    """
    Crea y compila un modelo liviano para la detección de accidentes en secuencias de video,
    optimizado para entrenamiento en CPU.
    """
    sequence_input = Input(shape=(Config.SEQUENCE_LENGTH, Config.IMG_SIZE, Config.IMG_SIZE, 3), name='video_input')
    
    # Base model: MobileNetV2 preentrenado en ImageNet (más ligero que EfficientNetB0)
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(Config.IMG_SIZE, Config.IMG_SIZE, 3)
    )
    base_model.trainable = False  # Congelamos el modelo base para reducir el cómputo
    
    # Extracción de características por frame
    x = TimeDistributed(base_model)(sequence_input)
    x = TimeDistributed(GlobalAveragePooling2D())(x)  # (batch, SEQUENCE_LENGTH, features)
    
    # Capa LSTM simple para capturar la información temporal
    x = LSTM(64, return_sequences=False)(x)
    
    # Capas densas para la clasificación final
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid', dtype='float32')(x)
    
    model = Model(inputs=sequence_input, outputs=outputs, name='accident_detector_video_light')
    
    if training:
        optimizer = tf.keras.optimizers.Adam(learning_rate=Config.INITIAL_LR)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    return model

# ==================== EVALUACIÓN DEL MODELO ====================
def evaluate_model(model, data_dir='data_car', results_dir=None):
    """
    Evalúa el modelo utilizando un generador basado en secuencias extraídas de videos.
    Genera reportes y visualizaciones (ROC, Precision-Recall y Matriz de Confusión) y los guarda en un directorio.
    """
    print("\nCargando datos de evaluación...")
    eval_generator = create_evaluation_sequence_generator(data_dir)
    
    print("\nRealizando predicciones...")
    predictions_proba = model.predict(eval_generator).flatten()
    predictions = (predictions_proba > Config.PRECISION_THRESHOLD).astype(int)
    true_labels = np.array(eval_generator.labels)
    
    print("\nCalculando métricas...")
    unique_classes = np.unique(np.concatenate([true_labels, predictions]))
    if len(unique_classes) == 1:
        target_names = ['Normal_Videos_for_Event_Recognition'] if 0 in unique_classes else ['CrashAccidents']
    else:
        target_names = ['Normal_Videos_for_Event_Recognition', 'CrashAccidents']
    
    report = classification_report(
        true_labels, predictions,
        target_names=target_names,
        output_dict=True
    )
    
    fpr, tpr, _ = roc_curve(true_labels, predictions_proba)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, _ = precision_recall_curve(true_labels, predictions_proba)
    pr_auc = auc(recall, precision)
    
    cm = confusion_matrix(true_labels, predictions)
    
    if results_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"evaluation_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(results_dir, 'roc_curve.png'))
    plt.close()
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(results_dir, 'pr_curve.png'))
    plt.close()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión')
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
    plt.close()
    
    with open(os.path.join(results_dir, 'metrics_report.txt'), 'w') as f:
        f.write("==================================================\n")
        f.write("REPORTE DE EVALUACIÓN DEL MODELO DE DETECCIÓN DE ACCIDENTES\n")
        f.write("==================================================\n\n")
        f.write("1. MÉTRICAS PRINCIPALES\n")
        f.write("-------------------------\n")
        f.write(f"Accuracy: {report['accuracy']:.4f}\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n")
        f.write(f"PR AUC: {pr_auc:.4f}\n\n")
        f.write("2. MÉTRICAS POR CLASE\n")
        f.write("-------------------------\n\n")
        for class_name in target_names:
            f.write(f"Clase: {class_name}\n")
            f.write(f"Precision: {report[class_name]['precision']:.4f}\n")
            f.write(f"Recall: {report[class_name]['recall']:.4f}\n")
            f.write(f"F1-score: {report[class_name]['f1-score']:.4f}\n\n")
        f.write("3. MATRIZ DE CONFUSIÓN\n")
        f.write("-------------------------\n")
        f.write("Filas: Verdadero, Columnas: Predicho\n")
        f.write(str(cm))
    
    print("\nRESUMEN DE RESULTADOS:")
    with open(os.path.join(results_dir, 'metrics_report.txt'), 'r') as f:
        print(f.read())
    
    print(f"\nEvaluación completada. Resultados guardados en: {results_dir}")
    
    return {
        'accuracy': report['accuracy'],
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm,
        'classification_report': report
    }

# ==================== ENTRENAMIENTO ====================
def train_model():
    """
    Función principal de entrenamiento adaptada para CPU con un modelo liviano:
      - Configura el entorno.
      - Genera los generadores de datos a partir de secuencias de videos.
      - Crea y entrena el modelo.
      - Evalúa el rendimiento y guarda los resultados.
    """
    print("\nConfigurando entrenamiento...")
    Config.setup()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"training_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Scheduler para disminuir el learning rate
    def scheduler(epoch, lr):
        return lr * 0.95
    lr_scheduler = LearningRateScheduler(scheduler, verbose=1)
    
    callbacks = [
        ModelCheckpoint(
            os.path.join(results_dir, 'best_model_accident_video_light.h5'),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            mode='max',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_accuracy',
            mode='max',
            factor=0.5,
            patience=3,
            min_lr=Config.MIN_LR,
            verbose=1
        ),
        lr_scheduler,
        TimeRemainingCallback()
    ]
    
    train_generator, validation_generator, class_weights = create_video_sequence_generators()
    model = create_model(training=True)
    
    print("\nIniciando entrenamiento del detector de accidentes (modelo liviano, CPU)...")
    try:
        history = model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=Config.EPOCHS,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        print("\nEvaluando el modelo...")
        evaluation_metrics = evaluate_model(model, data_dir='data_car', results_dir=results_dir)
        
        return model, history, results_dir, evaluation_metrics
        
    except Exception as e:
        print(f"\nError durante el entrenamiento: {e}")
        raise e

# ==================== (Opcional) INFERENCIA EN TIEMPO REAL ====================
def preprocess_frame(frame):
    """
    Preprocesa un frame para la inferencia en tiempo real:  
      - Convierte BGR a RGB  
      - Redimensiona al tamaño requerido  
      - Aplica el preprocesamiento de MobileNetV2
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (Config.IMG_SIZE, Config.IMG_SIZE))
    frame = preprocess_input(frame)
    return frame

def trigger_alert():
    """
    Función placeholder para activar una alerta (por ejemplo, enviar una notificación).
    """
    print("¡Alerta de accidente detectada!")

def real_time_detection(camera_source=0):
    """
    Realiza detección en tiempo real usando el modelo entrenado.  
    Se captura video desde la cámara (o fuente de video), se preprocesan los frames,  
    se acumula en un buffer y cuando se tiene una secuencia completa se realiza la predicción.
    """
    model = tf.keras.models.load_model('best_model_accident_video_light.h5')
    buffer = []
    
    cap = cv2.VideoCapture(camera_source)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed = preprocess_frame(frame)
        buffer.append(processed)
        
        if len(buffer) == Config.SEQUENCE_LENGTH:
            prediction = model.predict(np.expand_dims(buffer, 0))
            if prediction[0][0] > Config.PRECISION_THRESHOLD:
                trigger_alert()
            buffer.pop(0)
    cap.release()

# ==================== BLOQUE PRINCIPAL ====================
if __name__ == "__main__":
    try:
        model, history, results_dir, evaluation_metrics = train_model()
        print("\n¡Entrenamiento y evaluación del detector de accidentes completados exitosamente!")
        # Para realizar inferencia en tiempo real, descomenta la siguiente línea:
        # real_time_detection(camera_source=0)
    except Exception as e:
        print(f"\nError en el proceso: {e}")