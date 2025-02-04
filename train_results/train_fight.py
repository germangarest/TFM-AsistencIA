import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import os
from datetime import datetime
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración optimizada para detección de peleas
class Config:
    # Parámetros del modelo
    IMG_SIZE = 128  # Tamaño reducido para rendimiento
    BATCH_SIZE = 32  # Balance entre velocidad y memoria
    EPOCHS = 30     
    INITIAL_LR = 1e-3
    MIN_LR = 1e-6
    VALIDATION_SPLIT = 0.2
    
    # Parámetros de entrenamiento
    USE_MIXED_PRECISION = True  # Para GPU
    USE_CLASS_WEIGHTS = True
    UNFREEZE_LAYERS = 20
    
    # Parámetros de aumento de datos (ajustados para detección de peleas)
    ROTATION_RANGE = 10  # Menos rotación ya que la orientación puede ser importante
    SHIFT_RANGE = 0.1
    ZOOM_RANGE = 0.1
    BRIGHTNESS_RANGE = [0.9, 1.1]
    
    @classmethod
    def setup(cls):
        # Optimizaciones de memoria y CPU/GPU
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            try:
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
                if cls.USE_MIXED_PRECISION:
                    policy = tf.keras.mixed_precision.Policy('mixed_float16')
                    tf.keras.mixed_precision.set_global_policy(policy)
                    print("\nMixed precision activada")
            except Exception as e:
                print(f"Error configurando GPU: {str(e)}")
        else:
            print("\nNo se detectó GPU, usando CPU")
            
        # Optimizaciones de rendimiento
        try:
            tf.config.optimizer.set_jit(True)  # XLA
            print("XLA optimización activada")
        except:
            print("XLA optimización no disponible")
        
        # Optimizar threads para CPU
        try:
            tf.config.threading.set_inter_op_parallelism_threads(4)
            tf.config.threading.set_intra_op_parallelism_threads(4)
            print("Optimización de threads configurada")
        except:
            print("No se pudo configurar optimización de threads")

def create_model(input_shape=(Config.IMG_SIZE, Config.IMG_SIZE, 3), training=True):
    """Modelo optimizado para detección de peleas"""
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape,
        alpha=0.75  # Versión ligera del modelo
    )
    
    # Fine-tuning selectivo
    base_model.trainable = True
    for layer in base_model.layers[:-Config.UNFREEZE_LAYERS]:
        layer.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=outputs, name='fight_detector')
    
    if training:
        optimizer = tf.keras.optimizers.Adam(learning_rate=Config.INITIAL_LR)
        if Config.USE_MIXED_PRECISION:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall')]
        )
    
    return model

def create_data_generators():
    """Generadores de datos optimizados para detección de peleas"""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=Config.ROTATION_RANGE,
        width_shift_range=Config.SHIFT_RANGE,
        height_shift_range=Config.SHIFT_RANGE,
        zoom_range=Config.ZOOM_RANGE,
        brightness_range=Config.BRIGHTNESS_RANGE,
        horizontal_flip=True,  # Las peleas pueden verse desde ambos lados
        fill_mode='nearest',
        validation_split=Config.VALIDATION_SPLIT,
        preprocessing_function=None  # Optimización de velocidad
    )

    valid_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=Config.VALIDATION_SPLIT
    )

    try:
        train_generator = train_datagen.flow_from_directory(
            'data',
            target_size=(Config.IMG_SIZE, Config.IMG_SIZE),
            batch_size=Config.BATCH_SIZE,
            class_mode='binary',
            classes=['NormalVideos', 'Fighting'],  # Clases adaptadas para peleas
            subset='training',
            shuffle=True,
            seed=42
        )

        validation_generator = valid_datagen.flow_from_directory(
            'data',
            target_size=(Config.IMG_SIZE, Config.IMG_SIZE),
            batch_size=Config.BATCH_SIZE,
            class_mode='binary',
            classes=['NormalVideos', 'Fighting'],  # Clases adaptadas para peleas
            subset='validation',
            shuffle=False,
            seed=42
        )
    except Exception as e:
        print(f"Error cargando datos: {str(e)}")
        raise e
    
    if Config.USE_CLASS_WEIGHTS:
        total = len(train_generator.classes)
        class_counts = np.bincount(train_generator.classes)
        class_weights = {
            0: total / (2 * class_counts[0]),  # NormalVideos
            1: total / (2 * class_counts[1])   # Fighting
        }
        print(f"\nPesos de clases calculados: {class_weights}")
    else:
        class_weights = None
    
    return train_generator, validation_generator, class_weights

def evaluate_model(model, data_dir='data', results_dir=None):
    """Evalúa el modelo y genera visualizaciones y métricas"""
    print("\nCargando datos de evaluación...")
    eval_datagen = ImageDataGenerator(rescale=1./255)
    
    eval_generator = eval_datagen.flow_from_directory(
        data_dir,
        target_size=(Config.IMG_SIZE, Config.IMG_SIZE),
        batch_size=Config.BATCH_SIZE,
        class_mode='binary',
        classes=['NormalVideos', 'Fighting'],
        shuffle=False
    )
    
    print("\nRealizando predicciones...")
    predictions_proba = model.predict(eval_generator)
    predictions = (predictions_proba > 0.5).astype(int)
    true_labels = eval_generator.classes
    
    # Calcular métricas
    print("\nCalculando métricas...")
    report = classification_report(true_labels, predictions, target_names=['Normal', 'Pelea'], output_dict=True)
    
    # ROC y PR curves
    fpr, tpr, _ = roc_curve(true_labels, predictions_proba)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, _ = precision_recall_curve(true_labels, predictions_proba)
    pr_auc = auc(recall, precision)
    
    # Matriz de confusión
    cm = confusion_matrix(true_labels, predictions)
    
    if results_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"evaluation_results_{timestamp}"
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Generar visualizaciones
    print("\nGenerando visualizaciones...")
    
    # ROC Curve
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
    
    # Precision-Recall Curve
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
    
    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión')
    plt.ylabel('Verdadero')
    plt.xlabel('Predicho')
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Guardar métricas en un archivo de texto
    with open(os.path.join(results_dir, 'metrics_report.txt'), 'w') as f:
        f.write("==================================================\n")
        f.write("REPORTE DE EVALUACIÓN DEL MODELO DE DETECCIÓN DE PELEAS\n")
        f.write("==================================================\n\n")
        
        f.write("1. MÉTRICAS PRINCIPALES\n")
        f.write("--------------------\n")
        f.write(f"Accuracy: {report['accuracy']:.4f}\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n")
        f.write(f"PR AUC: {pr_auc:.4f}\n\n")
        
        f.write("2. MÉTRICAS POR CLASE\n")
        f.write("--------------------\n\n")
        for class_name in ['Normal', 'Pelea']:
            f.write(f"Clase: {class_name}\n")
            f.write(f"Precision: {report[class_name.lower()]['precision']:.4f}\n")
            f.write(f"Recall: {report[class_name.lower()]['recall']:.4f}\n")
            f.write(f"F1-score: {report[class_name.lower()]['f1-score']:.4f}\n\n")
        
        f.write("3. MATRIZ DE CONFUSIÓN\n")
        f.write("--------------------\n")
        f.write("Filas: Verdadero, Columnas: Predicho\n")
        f.write(str(cm))
    
    # Imprimir resumen
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

def train_model():
    """Entrenamiento optimizado para detección de peleas"""
    print("\nConfigurando entrenamiento...")
    Config.setup()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"training_results_fight_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            os.path.join(results_dir, 'best_model_fight.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=3,
            min_lr=Config.MIN_LR,
            verbose=1
        )
    ]
    
    train_generator, validation_generator, class_weights = create_data_generators()
    model = create_model(training=True)
    
    print("\nIniciando entrenamiento del detector de peleas...")
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
        evaluation_metrics = evaluate_model(model, 'data', results_dir)
        
        return model, history, results_dir, evaluation_metrics
        
    except Exception as e:
        print(f"\nError durante el entrenamiento: {str(e)}")
        raise e

if __name__ == "__main__":
    try:
        model, history, results_dir, evaluation_metrics = train_model()
        print(f"\nEntrenamiento y evaluación del detector de peleas completados exitosamente!")
    except Exception as e:
        print(f"\nError en el proceso: {str(e)}")