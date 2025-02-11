import os
import cv2
import numpy as np
from datetime import datetime
from PIL import Image
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms.functional as TF

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve, confusion_matrix

# ------------------------- FUNCIONES AUXILIARES -------------------------
def format_time(seconds):
    """Formatea segundos a HH:MM:SS."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

# ------------------------- CONFIGURACIÓN -------------------------
class Config:
    # Parámetros del dataset y entrenamiento adaptados para CPU
    DATA_DIR = 'data_fight'       # Carpeta de datos
    IMG_SIZE = 64                 # Tamaño reducido de imagen (ancho y alto)
    BATCH_SIZE = 4                # Tamaño de lote reducido
    EPOCHS = 5                    # Menos épocas para entrenamiento rápido
    INITIAL_LR = 1e-4             # Learning rate inicial
    MIN_LR = 1e-6                 # LR mínimo (para el scheduler)
    VALIDATION_SPLIT = 0.2        # Fracción de datos para validación
    PRECISION_THRESHOLD = 0.6     # Umbral para clasificar las predicciones

    # Parámetros de entrenamiento
    USE_MIXED_PRECISION = False   # Desactivado, optimizado para CPU
    USE_CLASS_WEIGHTS = True      # Balanceo de clases
    NUM_WORKERS = 1               # Menos procesos para cargar datos en CPU

    # Parámetros de Data Augmentation (se mantienen, pero se pueden desactivar si se desea)
    ROTATION_RANGE = 10         
    SHIFT_RANGE = 0.1
    BRIGHTNESS_RANGE = [0.85, 1.15]
    HORIZONTAL_FLIP = True

    # Parámetros para secuencias de video
    SEQUENCE_LENGTH = 5           # Menos frames en la secuencia
    FRAME_STRIDE = 2              # Stride para la ventana deslizante

    @classmethod
    def setup(cls):
        # Forzar el uso de CPU
        cls.device = torch.device('cpu')
        print(f"Usando dispositivo: {cls.device}")
        cls.USE_MIXED_PRECISION = False

# ------------------------- LECTURA Y GENERACIÓN DE SECUENCIAS -------------------------
def read_video_sequence(video_path, start_frame, sequence_length, target_size):
    """
    Lee una secuencia de frames desde un video, iniciando en start_frame.
    Si no se pueden leer suficientes frames, se repite el último.
    Retorna una lista de imágenes PIL (RGB) redimensionadas.
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
        # Convertir de BGR a RGB y redimensionar
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
        frames.append(Image.fromarray(frame))
    cap.release()
    if len(frames) == 0:
        raise ValueError(f"No se pudieron leer frames del video: {video_path}")
    while len(frames) < sequence_length:
        frames.append(frames[-1])
    return frames

def generate_video_sequences(video_dir, sequence_length, stride):
    """
    A partir de un directorio con archivos de video, genera una lista de secuencias.
    Cada secuencia se representa como una tupla: (ruta_del_video, frame_inicial).
    Se utiliza una ventana deslizante con el stride indicado.
    """
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    sequences = []
    video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir)
                   if f.lower().endswith(video_extensions)]
    video_files.sort()
    for video_file in video_files:
        print(f"Procesando video: {os.path.basename(video_file)}")
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"Advertencia: No se pudo abrir {video_file}. Se omite.")
            continue
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if total_frames < sequence_length:
            print(f"Advertencia: {video_file} tiene menos frames ({total_frames}) que SEQUENCE_LENGTH ({sequence_length}). Se omite.")
            continue
        for start in range(0, total_frames - sequence_length + 1, stride):
            sequences.append((video_file, start))
    return sequences

# ------------------------- DATASET DE SECUENCIAS DE VIDEO -------------------------
class VideoDataset(Dataset):
    """
    Dataset que extrae secuencias de frames desde archivos de video.
    Cada muestra es una secuencia de frames (tensor de forma (C, T, H, W)) y su etiqueta.
    """
    def __init__(self, samples, labels, target_size, sequence_length, augment=False):
        self.samples = samples
        self.labels = labels
        self.target_size = target_size
        self.sequence_length = sequence_length
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, start_frame = self.samples[idx]
        frames = read_video_sequence(video_path, start_frame, self.sequence_length, self.target_size)
        if self.augment:
            # Data augmentation sencilla: rotación, traslación, flip y ajuste de brillo
            angle = np.random.uniform(-Config.ROTATION_RANGE, Config.ROTATION_RANGE)
            max_dx = Config.SHIFT_RANGE * self.target_size[0]
            max_dy = Config.SHIFT_RANGE * self.target_size[1]
            translate = (np.random.uniform(-max_dx, max_dx), np.random.uniform(-max_dy, max_dy))
            brightness_factor = np.random.uniform(Config.BRIGHTNESS_RANGE[0], Config.BRIGHTNESS_RANGE[1])
            # Se aplica a cada frame
            frames = [TF.affine(frame, angle=angle, translate=translate, scale=1.0, shear=0,
                                  interpolation=TF.InterpolationMode.BILINEAR, fill=0) for frame in frames]
            if Config.HORIZONTAL_FLIP and np.random.rand() < 0.5:
                frames = [TF.hflip(frame) for frame in frames]
            frames = [TF.adjust_brightness(frame, brightness_factor) for frame in frames]
        # Convertir cada frame a tensor y normalizar a [-1,1]
        frames_tensor = [TF.to_tensor(frame) for frame in frames]
        frames_tensor = [frame * 2 - 1 for frame in frames_tensor]
        video_tensor = torch.stack(frames_tensor, dim=0).permute(1, 0, 2, 3)  # (C, T, H, W)
        label = self.labels[idx]
        return video_tensor, label

def create_video_sequence_datasets(data_dir=Config.DATA_DIR):
    """
    Crea datasets de entrenamiento y validación a partir de dos subcarpetas en 'data_fight':
      - 'Normal_Videos_for_Event_Recognition'
      - 'Fighting'
    """
    classes = ['Normal_Videos_for_Event_Recognition', 'Fighting']
    train_samples, train_labels = [], []
    val_samples, val_labels = [], []
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

    train_dataset = VideoDataset(train_samples, train_labels, (Config.IMG_SIZE, Config.IMG_SIZE),
                                 Config.SEQUENCE_LENGTH, augment=True)
    val_dataset = VideoDataset(val_samples, val_labels, (Config.IMG_SIZE, Config.IMG_SIZE),
                               Config.SEQUENCE_LENGTH, augment=False)

    # Calcular pesos de clases para balancear (opcional)
    if Config.USE_CLASS_WEIGHTS:
        total = len(train_labels)
        counts = np.bincount(train_labels)
        class_weights = {
            0: total / (2 * counts[0]) if counts[0] > 0 else 1.0,
            1: total / (2 * counts[1]) if counts[1] > 0 else 1.0
        }
        print(f"Pesos de clases: {class_weights}")
    else:
        class_weights = None

    return train_dataset, val_dataset, class_weights

def create_evaluation_dataset(data_dir=Config.DATA_DIR):
    """
    Crea un dataset para evaluación (sin data augmentation) a partir de los videos.
    """
    classes = ['Normal_Videos_for_Event_Recognition', 'Fighting']
    samples, labels = [], []
    stride = Config.FRAME_STRIDE

    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(data_dir, class_name)
        sequences = generate_video_sequences(class_path, Config.SEQUENCE_LENGTH, stride)
        samples.extend(sequences)
        labels.extend([class_idx] * len(sequences))
    
    eval_dataset = VideoDataset(samples, labels, (Config.IMG_SIZE, Config.IMG_SIZE),
                                Config.SEQUENCE_LENGTH, augment=False)
    return eval_dataset

# ------------------------- MODELO LIGERO PARA VIDEO -------------------------
def get_model():
    """
    Crea un modelo ligero basado en ResNet18 para clasificación de video.
    Se procesa cada frame individualmente y se promedian las características temporales.
    La entrada esperada es (B, 3, T, H, W).
    """
    class SimpleVideoClassifier(nn.Module):
        def __init__(self, num_classes=1):
            super(SimpleVideoClassifier, self).__init__()
            # Se utiliza ResNet18 preentrenada para extraer características 2D
            self.resnet = torchvision.models.resnet18(pretrained=True)
            self.resnet.fc = nn.Identity()  # Remover la capa de clasificación original
            self.fc = nn.Linear(512, num_classes)
        
        def forward(self, x):
            # x: (B, 3, T, H, W)
            B, C, T, H, W = x.shape
            # Reorganizar a (B*T, C, H, W) para procesar cada frame por separado
            x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
            features = self.resnet(x)  # (B*T, 512)
            # Reorganizar a (B, T, 512)
            features = features.view(B, T, -1)
            # Promediar las características en la dimensión temporal
            features = features.mean(dim=1)
            out = self.fc(features)  # (B, 1)
            return out

    return SimpleVideoClassifier().to(Config.device)

# ------------------------- EVALUACIÓN DEL MODELO -------------------------
def evaluate_model(model, device, results_dir, eval_loader):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    print("Iniciando evaluación del modelo...")
    with torch.no_grad():
        for batch_idx, (videos, labels) in enumerate(eval_loader):
            videos = videos.to(device)
            labels = labels.to(device)
            outputs = model(videos).squeeze(1)
            probs = torch.sigmoid(outputs)
            preds = (probs > Config.PRECISION_THRESHOLD).long()
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(all_labels, all_preds, output_dict=True)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    precision_vals, recall_vals, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(recall_vals, precision_vals)
    cm = confusion_matrix(all_labels, all_preds)

    os.makedirs(results_dir, exist_ok=True)
    # Guardar gráficos
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, lw=2, label=f'ROC AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'roc_curve.png'))
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.plot(recall_vals, precision_vals, lw=2, label=f'PR AUC = {pr_auc:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'pr_curve.png'))
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicción')
    plt.ylabel('Etiqueta')
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
    plt.close()

    report_path = os.path.join(results_dir, 'metrics_report.txt')
    with open(report_path, 'w') as f:
        f.write("REPORTE DE EVALUACIÓN DEL MODELO\n")
        f.write("====================================\n\n")
        f.write(f"Accuracy: {report['accuracy']:.4f}\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n")
        f.write(f"PR AUC: {pr_auc:.4f}\n\n")
        f.write("Métricas por clase:\n")
        for key, value in report.items():
            if key in ['accuracy', 'macro avg', 'weighted avg']:
                continue
            f.write(f"Clase {key} -> Precision: {value['precision']:.4f}, Recall: {value['recall']:.4f}, F1: {value['f1-score']:.4f}\n")
    with open(report_path, 'r') as f:
        print(f.read())

    return {
        'accuracy': report['accuracy'],
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm,
        'classification_report': report
    }

# ------------------------- CICLO DE ENTRENAMIENTO -------------------------
def train_model():
    print("\nConfigurando entrenamiento...")
    Config.setup()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"training_results_video_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # Crear datasets y dataloaders
    train_dataset, val_dataset, class_weights = create_video_sequence_datasets()
    print(f"Dataset de entrenamiento: {len(train_dataset)} muestras")
    print(f"Dataset de validación: {len(val_dataset)} muestras")
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True,
                              num_workers=Config.NUM_WORKERS, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
                            num_workers=Config.NUM_WORKERS, pin_memory=False)

    model = get_model()
    
    # Definir pérdida y optimizador
    if class_weights is not None:
        pos_weight = torch.tensor(class_weights[0] / class_weights[1], device=Config.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = AdamW(model.parameters(), lr=Config.INITIAL_LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR)

    scaler = None  # No se usa precisión mixta en CPU

    best_pr_auc = 0
    patience = 5
    epochs_without_improvement = 0
    train_history = {'loss': [], 'val_loss': [], 'val_pr_auc': []}
    epoch_durations = []

    print("\n===== INICIANDO ENTRENAMIENTO =====")
    for epoch in range(Config.EPOCHS):
        print(f"\n========== ÉPOCA {epoch+1}/{Config.EPOCHS} ==========")
        print(f"Learning rate actual: {optimizer.param_groups[0]['lr']:.6f}")
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        total_batches = len(train_loader)

        for batch_idx, (videos, labels) in enumerate(train_loader):
            if (batch_idx + 1) % 5 == 0 or batch_idx == 0:
                current_batch = batch_idx + 1
                elapsed = time.time() - epoch_start_time
                estimated_total = (elapsed / current_batch) * total_batches
                epoch_remaining = estimated_total - elapsed
                avg_epoch = np.mean(epoch_durations) if epoch_durations else estimated_total
                total_remaining = epoch_remaining + avg_epoch * (Config.EPOCHS - (epoch + 1))
                print(f"[Época {epoch+1}] Batch {current_batch}/{total_batches} | "
                      f"Transcurrido: {format_time(elapsed)} | "
                      f"Restante en época: {format_time(epoch_remaining)} | "
                      f"Total restante estimado: {format_time(total_remaining)}")
            
            videos = videos.to(Config.device)
            labels = labels.float().to(Config.device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(videos).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * videos.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)

        # Validación
        model.eval()
        val_loss = 0.0
        all_val_probs = []
        all_val_labels = []
        print(f"\n[Época {epoch+1}] Iniciando validación...")
        with torch.no_grad():
            for videos, labels in val_loader:
                videos = videos.to(Config.device)
                labels = labels.float().to(Config.device)
                outputs = model(videos).squeeze(1)
                loss = criterion(outputs, labels)
                probs = torch.sigmoid(outputs)
                val_loss += loss.item() * videos.size(0)
                all_val_probs.extend(probs.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
        val_loss = val_loss / len(val_loader.dataset)
        precision_vals, recall_vals, _ = precision_recall_curve(all_val_labels, all_val_probs)
        val_pr_auc = auc(recall_vals, precision_vals)

        epoch_duration = time.time() - epoch_start_time
        epoch_durations.append(epoch_duration)
        print(f"\n========== RESUMEN ÉPOCA {epoch+1} ==========")
        print(f"Duración: {format_time(epoch_duration)}")
        print(f"Pérdida entrenamiento: {epoch_loss:.4f}")
        print(f"Pérdida validación: {val_loss:.4f}")
        print(f"Val PR AUC: {val_pr_auc:.4f}")
        print("=============================================")
        train_history['loss'].append(epoch_loss)
        train_history['val_loss'].append(val_loss)
        train_history['val_pr_auc'].append(val_pr_auc)

        if val_pr_auc > best_pr_auc:
            best_pr_auc = val_pr_auc
            epochs_without_improvement = 0
            best_model_path = os.path.join(results_dir, 'best_model_fight_video.pth')
            torch.save(model.state_dict(), best_model_path)
            print("¡Mejor modelo guardado!")
        else:
            epochs_without_improvement += 1
            print(f"No se mejoró el PR AUC en la época {epoch+1} (paciencia: {epochs_without_improvement}/{patience})")
            if epochs_without_improvement >= patience:
                print("Early stopping activado. Finalizando entrenamiento.")
                break

        scheduler.step()

    print("\n===== ENTRENAMIENTO COMPLETADO =====")
    print("Iniciando evaluación final del modelo...\n")
    # Crear loader para evaluación
    eval_dataset = create_evaluation_dataset()
    eval_loader = DataLoader(eval_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
                             num_workers=Config.NUM_WORKERS, pin_memory=False)
    evaluation_metrics = evaluate_model(model, Config.device, results_dir, eval_loader)

    return model, train_history, results_dir, evaluation_metrics

# ------------------------- BLOQUE PRINCIPAL -------------------------
if __name__ == "__main__":
    try:
        model, history, results_dir, evaluation_metrics = train_model()
        print("\n¡Entrenamiento y evaluación completados exitosamente!")
    except Exception as e:
        print(f"\nError en el proceso: {e}")