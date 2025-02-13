import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import os
import time
import gc
import torch
import torch.nn as nn

# ====================================================
# REGISTRO DE OBJETOS PERSONALIZADOS PARA KERAS
# ====================================================
tf.keras.utils.get_custom_objects()['Sequential'] = tf.keras.models.Sequential
from tensorflow.keras.layers import TimeDistributed as OriginalTimeDistributed

class FixedTimeDistributed(OriginalTimeDistributed):
    def __init__(self, *args, **kwargs):
        super(FixedTimeDistributed, self).__init__(*args, **kwargs)
        self._self_tracked_trackables = {}

tf.keras.utils.get_custom_objects()['TimeDistributed'] = FixedTimeDistributed

from tensorflow.keras.layers import BatchNormalization as OriginalBatchNormalization

@tf.keras.utils.register_keras_serializable()
class CustomBatchNormalization(OriginalBatchNormalization):
    pass

try:
    from tensorflow.python.keras.engine.functional import Functional
    tf.keras.utils.get_custom_objects()['Functional'] = Functional
except ImportError:
    pass

from tensorflow.keras.layers import InputLayer as _InputLayer
class FixedInputLayer(_InputLayer):
    def __init__(self, **kwargs):
        if 'batch_shape' in kwargs:
            kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
        super(FixedInputLayer, self).__init__(**kwargs)
        self._self_tracked_trackables = {}

from tensorflow.keras.layers import BatchNormalization
tf.keras.utils.get_custom_objects()['BatchNormalization'] = BatchNormalization

# ====================================================
# CONSTANTES Y CONFIGURACIÓN
# ====================================================
ACCIDENT_IMG_SIZE = 160   # Para el modelo de accidentes
FIGHT_IMG_SIZE = 64       # Para el modelo de peleas
FIRE_IMG_SIZE = 128       # Para el modelo de incendios
SEQUENCE_LENGTH = 5       # Número de frames por secuencia
ACCIDENT_THRESHOLD = 0.90
FIGHT_THRESHOLD = 0.90
FIRE_THRESHOLD = 0.90
SMOOTHING_WINDOW = 3
DISPLAY_FPS = 30          # FPS deseado para visualización

st.set_page_config(page_title="AsistencIA - Detección de Incidentes", 
                   page_icon="🚨", layout="wide")

# ====================================================
# ESTILOS CSS PERSONALIZADOS
# ====================================================
st.markdown("""
<style>
    body { background-color: #f4f4f4; }
    .big-font { font-size: 24px !important; font-weight: bold; }
    .report-font { font-size: 18px !important; }
    .stProgress > div > div > div > div {background-color: #ff4c4c;}
    .streamlit-expanderHeader { font-weight: bold; font-size: 18px; }
    .webcam-container { border: 2px solid #ddd; border-radius: 5px; padding: 10px; margin-bottom: 20px; background: #fff; }
    .alert-active { animation: pulse 1s infinite alternate; }
    @keyframes pulse { from { background-color: rgba(255,0,0,0.3); } to { background-color: rgba(255,0,0,0.7); } }
    .unique-alert { background-color: rgba(255, 255, 0, 0.8); color: black; padding: 5px 10px; border-radius: 5px; font-size: 1em; margin: 5px 0; }
    .header-box {
        background-color: #0099cc;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        margin-bottom: 20px;
        font-size: 2em;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ====================================================
# CONFIGURACIÓN DE GPU (OPCIONAL)
# ====================================================
# En Streamlit Cloud es posible que no se disponga de GPU.
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Si deseas limitar la memoria, puedes hacerlo; de lo contrario, puedes omitirlo
        # tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
    except RuntimeError as e:
        st.error(e)

# ====================================================
# REGISTRO GLOBAL DE OBJETOS PERSONALIZADOS
# ====================================================
custom_objects = {
    'DTypePolicy': tf.keras.mixed_precision.Policy,
    'InputLayer': FixedInputLayer,
    'BatchNormalization': CustomBatchNormalization,
    'BatchNormalizationV2': CustomBatchNormalization,
    'TimeDistributed': FixedTimeDistributed     
}
tf.keras.utils.get_custom_objects().update(custom_objects)

# ====================================================
# CARGA DE MODELOS (CACHÉ)
# ====================================================
@st.cache_resource(show_spinner=False)
def load_models():
    with tf.device('/GPU:0' if gpus else '/CPU:0'):
        # --- Modelo de Accidentes (TensorFlow) ---
        accident_model = tf.keras.models.load_model('models/model_car.h5', compile=False, custom_objects=custom_objects)
        accident_model.compile(jit_compile=True)
        # Dummy para compilar el grafo (puedes reducir el tamaño si es necesario)
        dummy_accident = tf.zeros((1, SEQUENCE_LENGTH, ACCIDENT_IMG_SIZE, ACCIDENT_IMG_SIZE, 3))
        accident_model(dummy_accident)
        
    # --- Modelo de Incendios (TensorFlow) ---
    fire_model = tf.keras.models.load_model('models/model_fire.h5', compile=False, custom_objects=custom_objects)
    fire_model.compile(jit_compile=True)
    dummy_fire = tf.zeros((1, FIRE_IMG_SIZE, FIRE_IMG_SIZE, 3))
    _ = fire_model(dummy_fire)
    
    # --- Modelo de Peleas (PyTorch) ---
    '''
    class SimpleVideoClassifier(nn.Module):
        def __init__(self, num_classes=1):
            super(SimpleVideoClassifier, self).__init__()
            self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
            self.resnet.fc = nn.Identity()
            self.fc = nn.Linear(512, num_classes)
        
        def forward(self, x):
            B, C, T, H, W = x.shape
            x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
            features = self.resnet(x)
            features = features.view(B, T, -1)
            features = features.mean(dim=1)
            out = self.fc(features)
            return out

    fight_model = SimpleVideoClassifier()
    fight_model.load_state_dict(torch.load('models/model_fight.pth', map_location=torch.device('cpu')))
    fight_model.eval()

    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fight_model.to(torch_device)
    '''
    
    return accident_model, fire_model, # fight_model, torch_device

try:
    accident_model, fire_model, fight_model, torch_device = load_models()
except Exception as e:
    st.error(f"Error al cargar los modelos: {e}")
    st.stop()

# ====================================================
# FUNCIONES AUXILIARES
# ====================================================
def resize_frame(frame, size=(640, 480)):
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

@tf.function(reduce_retracing=True)
def predict_batch(model, input_tensor):
    return model(input_tensor, training=False)

def draw_overlays(frame, accident_prob, fight_prob, fire_prob):
    if accident_prob > ACCIDENT_THRESHOLD:
        cv2.putText(frame, "POSIBLE ACCIDENTE", (10, frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if fight_prob > FIGHT_THRESHOLD:
        cv2.putText(frame, "POSIBLE PELEA", (frame.shape[1] - 250, frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if fire_prob > FIRE_THRESHOLD:
        cv2.putText(frame, "POSIBLE INCENDIO", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return frame

def process_frame(frame, state):
    # Preprocesamiento de cada modelo
    resized_accident = cv2.resize(frame, (ACCIDENT_IMG_SIZE, ACCIDENT_IMG_SIZE), interpolation=cv2.INTER_AREA)
    processed_accident = np.float32(resized_accident) / 255.0
    
    resized_fight = cv2.resize(frame, (FIGHT_IMG_SIZE, FIGHT_IMG_SIZE), interpolation=cv2.INTER_AREA)
    processed_fight = np.float32(resized_fight) / 255.0
    
    resized_fire = cv2.resize(frame, (FIRE_IMG_SIZE, FIRE_IMG_SIZE), interpolation=cv2.INTER_AREA)
    processed_fire = np.float32(resized_fire) / 255.0

    # Inicialización y actualización de buffers
    if 'accident_buffer' not in state:
        state['accident_buffer'] = []
    if 'fight_buffer' not in state:
        state['fight_buffer'] = []
    if "last_accident_event_time" not in state:
        state["last_accident_event_time"] = None
    if "last_fight_event_time" not in state:
        state["last_fight_event_time"] = None
    if "last_fire_event_time" not in state:
        state["last_fire_event_time"] = None

    state['accident_buffer'].append(processed_accident)
    state['fight_buffer'].append(processed_fight)
    state['accident_buffer'] = state['accident_buffer'][-SEQUENCE_LENGTH:]
    state['fight_buffer'] = state['fight_buffer'][-SEQUENCE_LENGTH:]
    
    accident_seq = (state['accident_buffer'] + [processed_accident] * SEQUENCE_LENGTH)[:SEQUENCE_LENGTH]
    fight_seq = (state['fight_buffer'] + [processed_fight] * SEQUENCE_LENGTH)[:SEQUENCE_LENGTH]
    
    # Preparación de tensores para cada modelo
    accident_input = tf.convert_to_tensor(np.expand_dims(np.array(accident_seq), axis=0))
    
    fight_input = np.array(fight_seq)
    fight_input = np.transpose(fight_input, (0, 3, 1, 2))
    fight_input = np.expand_dims(fight_input, axis=0)
    fight_input = np.transpose(fight_input, (0, 2, 1, 3, 4))
    fight_input = torch.tensor(fight_input, dtype=torch.float32).to(torch_device)
    
    fire_input = tf.convert_to_tensor(np.expand_dims(processed_fire, axis=0))
    
    # Realización de predicciones
    accident_pred = predict_batch(accident_model, accident_input)
    accident_prob = float(accident_pred[0][0])
    
    with torch.no_grad():
        fight_output = fight_model(fight_input)
        fight_prob = torch.sigmoid(fight_output).item()
    
    fire_pred = fire_model(fire_input, training=False)
    fire_prob = float(fire_pred[0][1])
    
    # Suavizado de predicciones
    for key in ['accident_predictions', 'fight_predictions', 'fire_predictions']:
        if key not in state:
            state[key] = []
    
    state['accident_predictions'] = (state['accident_predictions'] + [accident_prob])[-SMOOTHING_WINDOW:]
    state['fight_predictions'] = (state['fight_predictions'] + [fight_prob])[-SMOOTHING_WINDOW:]
    state['fire_predictions'] = (state['fire_predictions'] + [fire_prob])[-SMOOTHING_WINDOW:]
    
    accident_smooth = sum(state['accident_predictions']) / len(state['accident_predictions'])
    fight_smooth = sum(state['fight_predictions']) / len(state['fight_predictions'])
    fire_smooth = sum(state['fire_predictions']) / len(state['fire_predictions'])
    
    state['last_accident_pred'] = accident_smooth
    state['last_fight_pred'] = fight_smooth
    state['last_fire_pred'] = fire_smooth

    out_frame = resize_frame(frame)
    out_frame = draw_overlays(out_frame, accident_smooth, fight_smooth, fire_smooth)
    
    return out_frame, accident_smooth, fight_smooth, fire_smooth

def add_detection_event(event_type, confidence, last_event_key):
    current_time = datetime.now()
    if (st.session_state.get(last_event_key) is None or 
        (current_time - st.session_state[last_event_key]).total_seconds() > 10):
        event = {
            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "video": st.session_state.get("current_video_name", "Webcam"),
            "type": event_type,
            "confidence": confidence
        }
        st.session_state["detection_history"].append(event)
        st.session_state[last_event_key] = current_time

# ====================================================
# INICIALIZACIÓN DEL ESTADO DE LA SESIÓN
# ====================================================
for key in ['detection_history', 'detected_events', 'consecutive_alerts_accident', 'alert_active_accident',
            'last_alert_time_accident', 'consecutive_alerts_fight', 'alert_active_fight', 'last_alert_time_fight',
            'current_video_name', 'webcam_active', 'last_accident_pred', 'last_fight_pred', 'last_fire_pred',
            'accident_predictions', 'fight_predictions', 'fire_predictions', 'accident_buffer', 'fight_buffer', 'run']:
    if key not in st.session_state:
        if key in ['detection_history']:
            st.session_state[key] = []
        elif key in ['detected_events']:
            st.session_state[key] = set()
        elif key in ['consecutive_alerts_accident', 'consecutive_alerts_fight']:
            st.session_state[key] = 0
        elif key in ['alert_active_accident', 'alert_active_fight']:
            st.session_state[key] = False
        elif key in ['last_alert_time_accident', 'last_alert_time_fight']:
            st.session_state[key] = datetime.now()
        elif key == 'current_video_name':
            st.session_state[key] = "Webcam"
        elif key == 'webcam_active':
            st.session_state[key] = False
        elif key in ['last_accident_pred', 'last_fight_pred', 'last_fire_pred']:
            st.session_state[key] = 0.0
        elif key in ['accident_predictions', 'fight_predictions', 'fire_predictions', 'accident_buffer', 'fight_buffer']:
            st.session_state[key] = []
        elif key == 'run':
            st.session_state[key] = False

# ====================================================
# INTERFAZ PRINCIPAL
# ====================================================
st.title("AsistencIA - Sistema de Alerta de Incidentes")
st.markdown("""
    <div class="header-box">
        <p>Proyecto diseñado para mejorar la seguridad pública mediante detección inteligente en cámaras de vigilancia.</p>
    </div>
    <p class="report-font">Opciones disponibles:</p>
    <ul class="report-font">
        <li><strong>Webcam:</strong> detección en tiempo real usando la cámara del dispositivo.</li>
        <li><strong>Analizar Video:</strong> subir un vídeo para evaluar el sistema.</li>
        <li><strong>Historial:</strong> registro de incidentes detectados.</li>
    </ul>
""", unsafe_allow_html=True)

active_tab = st.radio("Selecciona una opción:", ["Webcam", "Analizar Video", "Historial"], horizontal=True, key="active_tab")

if active_tab != "Webcam":
    st.session_state['run'] = False
if active_tab != "Analizar Video":
    st.session_state.pop("video_uploader", None)

# ====================================================
# PESTAÑA: WEBCAM
# ====================================================
if active_tab == "Webcam":
    st.header("Webcam")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        run = st.checkbox('Activar Webcam', key='run')
        video_placeholder = st.empty()
    
    with col2:
        st.markdown("### Predicciones")
        accident_placeholder = st.empty()
        fight_placeholder = st.empty()
        fire_placeholder = st.empty()
        alert_placeholder = st.empty()
    
    if run:
        st.session_state["current_video_name"] = "Webcam"
        st.session_state["accident_buffer"] = []
        st.session_state["fight_buffer"] = []
        st.session_state["accident_predictions"] = []
        st.session_state["fight_predictions"] = []
        st.session_state["fire_predictions"] = []
    
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, DISPLAY_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Aumentamos el intervalo para reducir la carga (por ejemplo, cada 5º frame)
        frame_skip_interval = 5
        frame_count = 0

        last_accident_prob = 0.0
        last_fight_prob = 0.0
        last_fire_prob = 0.0

        while st.session_state["run"]:
            frame_start = time.time()
            ret, frame = cap.read()
            if not ret:
                st.error("Error al acceder a la webcam")
                break

            frame_count += 1

            if frame_count % frame_skip_interval == 0:
                processed_frame, accident_prob, fight_prob, fire_prob = process_frame(frame, st.session_state)
                last_accident_prob = accident_prob
                last_fight_prob = fight_prob
                last_fire_prob = fire_prob
                out_frame = processed_frame  # Ya tiene overlay
            else:
                out_frame = resize_frame(frame)
                out_frame = draw_overlays(out_frame, last_accident_prob, last_fight_prob, last_fire_prob)
            
            video_placeholder.image(out_frame, channels="BGR")
            
            # Actualizar paneles de predicción
            accident_placeholder.markdown(
                f"""
                <div style='background-color: {"#ff4c4c" if last_accident_prob > ACCIDENT_THRESHOLD else "#4CAF50"}; 
                          padding: 10px; border-radius: 5px; color: white;'>
                    🚗 Probabilidad de Accidente: {last_accident_prob:.1%}
                </div>
                """, unsafe_allow_html=True)
            fight_placeholder.markdown(
                f"""
                <div style='background-color: {"#ff4c4c" if last_fight_prob > FIGHT_THRESHOLD else "#4CAF50"}; 
                          padding: 10px; border-radius: 5px; color: white;'>
                    👊 Probabilidad de Pelea: {last_fight_prob:.1%}
                </div>
                """, unsafe_allow_html=True)
            fire_placeholder.markdown(
                f"""
                <div style='background-color: {"#ff4c4c" if last_fire_prob > FIRE_THRESHOLD else "#4CAF50"}; 
                          padding: 10px; border-radius: 5px; color: white;'>
                    🔥 Probabilidad de Incendio: {last_fire_prob:.1%}
                </div>
                """, unsafe_allow_html=True)
            
            if (last_accident_prob > ACCIDENT_THRESHOLD or 
                last_fight_prob > FIGHT_THRESHOLD or 
                last_fire_prob > FIRE_THRESHOLD):
                alert_placeholder.markdown(
                    """
                    <div style='background-color: #ff4c4c; padding: 10px; 
                             border-radius: 5px; color: white; text-align: center;'>
                        ⚠️ ¡ALERTA! Se ha detectado una situación de riesgo
                    </div>
                    """, unsafe_allow_html=True)
            else:
                alert_placeholder.empty()
            
            if last_accident_prob > ACCIDENT_THRESHOLD:
                add_detection_event("Accidente", last_accident_prob, "last_accident_event_time")
            if last_fight_prob > FIGHT_THRESHOLD:
                add_detection_event("Pelea", last_fight_prob, "last_fight_event_time")
            if last_fire_prob > FIRE_THRESHOLD:
                add_detection_event("Incendio", last_fire_prob, "last_fire_event_time")

            elapsed = time.time() - frame_start
            time.sleep(max(0, (1 / DISPLAY_FPS) - elapsed))
        
        cap.release()
        gc.collect()

# ====================================================
# PESTAÑA: ANALIZAR VIDEO
# ====================================================
elif active_tab == "Analizar Video":
    st.header("Análisis de Video")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        video_file = st.file_uploader("Sube un video", type=["mp4", "avi", "mov"], key="video_uploader")
        placeholder_vid = st.empty()
    
    with col2:
        st.markdown("### Predicciones")
        accident_placeholder = st.empty()
        fight_placeholder = st.empty()
        fire_placeholder = st.empty()
        alert_placeholder = st.empty()
    
    if video_file is not None:
        st.session_state["current_video_name"] = video_file.name
        st.session_state["accident_buffer"] = []
        st.session_state["fight_buffer"] = []
        st.session_state["accident_predictions"] = []
        st.session_state["fight_predictions"] = []
        st.session_state["fire_predictions"] = []
    
        temp_dir = "temp_videos"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        file_extension = os.path.splitext(video_file.name)[1]
        temp_path = os.path.join(temp_dir, f"temp_video{file_extension}")
        
        try:
            with open(temp_path, "wb") as f:
                f.write(video_file.read())
            
            cap = cv2.VideoCapture(temp_path)
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            if not video_fps or video_fps <= 0:
                video_fps = DISPLAY_FPS

            frame_skip_interval = 5  # Incrementa este valor si es necesario
            frame_count = 0
            last_accident_prob = 0.0
            last_fight_prob = 0.0
            last_fire_prob = 0.0
            
            start_time = time.time()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                video_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                elapsed = time.time() - start_time
                if video_timestamp > elapsed:
                    time.sleep(video_timestamp - elapsed)
                
                if frame_count % frame_skip_interval == 0:
                    processed_frame, accident_prob, fight_prob, fire_prob = process_frame(frame, st.session_state)
                    last_accident_prob = accident_prob
                    last_fight_prob = fight_prob
                    last_fire_prob = fire_prob
                    out_frame = processed_frame
                else:
                    out_frame = resize_frame(frame)
                    out_frame = draw_overlays(out_frame, last_accident_prob, last_fight_prob, last_fire_prob)
                
                placeholder_vid.image(out_frame, channels="BGR")
                
                accident_placeholder.markdown(
                    f"""
                    <div style='background-color: {"#ff4c4c" if last_accident_prob > ACCIDENT_THRESHOLD else "#4CAF50"}; 
                              padding: 10px; border-radius: 5px; color: white;'>
                        🚗 Probabilidad de Accidente: {last_accident_prob:.1%}
                    </div>
                    """, unsafe_allow_html=True)
                fight_placeholder.markdown(
                    f"""
                    <div style='background-color: {"#ff4c4c" if last_fight_prob > FIGHT_THRESHOLD else "#4CAF50"}; 
                              padding: 10px; border-radius: 5px; color: white;'>
                        👊 Probabilidad de Pelea: {last_fight_prob:.1%}
                    </div>
                    """, unsafe_allow_html=True)
                fire_placeholder.markdown(
                    f"""
                    <div style='background-color: {"#ff4c4c" if last_fire_prob > FIRE_THRESHOLD else "#4CAF50"}; 
                              padding: 10px; border-radius: 5px; color: white;'>
                        🔥 Probabilidad de Incendio: {last_fire_prob:.1%}
                    </div>
                    """, unsafe_allow_html=True)
                
                if (last_accident_prob > ACCIDENT_THRESHOLD or 
                    last_fight_prob > FIGHT_THRESHOLD or 
                    last_fire_prob > FIRE_THRESHOLD):
                    alert_placeholder.markdown(
                        """
                        <div style='background-color: #ff4c4c; padding: 10px; 
                                 border-radius: 5px; color: white; text-align: center;'>
                            ⚠️ ¡ALERTA! Se ha detectado una situación de riesgo
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    alert_placeholder.empty()
                
                if last_accident_prob > ACCIDENT_THRESHOLD:
                    add_detection_event("Accidente", last_accident_prob, "last_accident_event_time")
                if last_fight_prob > FIGHT_THRESHOLD:
                    add_detection_event("Pelea", last_fight_prob, "last_fight_event_time")
                if last_fire_prob > FIRE_THRESHOLD:
                    add_detection_event("Incendio", last_fire_prob, "last_fire_event_time")
            
            cap.release()
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            try:
                os.rmdir(temp_dir)
            except OSError:
                pass

# ====================================================
# PESTAÑA: HISTORIAL
# ====================================================
elif active_tab == "Historial":
    st.header("Historial de Detecciones")
    
    detection_history = st.session_state.get("detection_history", [])
    
    if detection_history and len(detection_history) > 0:
        try:
            df = pd.DataFrame(detection_history)
            required_columns = {"video", "timestamp", "type", "confidence"}
            if not required_columns.issubset(df.columns):
                st.error("La estructura de las detecciones no es la esperada. Revisa el formato de los datos.")
            else:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce').dt.strftime("%d-%m-%Y %H:%M:%S")
                df = df.sort_values(by=["video", "timestamp"])
                
                st.subheader("Resumen por Video")
                video_summary = df.groupby("video").agg(
                    Eventos_Detectados=("type", lambda x: ', '.join(sorted(set(x)))),
                    Confianza_Maxima=("confidence", "max")
                ).reset_index()
                st.dataframe(video_summary, hide_index=True, use_container_width=True)
                
                st.subheader("Detalle de Detecciones")
                df_detalle = df[["timestamp", "video", "type", "confidence"]].rename(
                    columns={
                        "timestamp": "Hora",
                        "video": "Video",
                        "type": "Tipo de Evento",
                        "confidence": "Confianza"
                    }
                )
                st.dataframe(df_detalle, hide_index=True, use_container_width=True)
        except Exception as e:
            st.error(f"Error al procesar el historial: {e}")
    else:
        st.write("No hay detecciones registradas aún.")
    
    if st.button("Limpiar Historial", use_container_width=True):
        st.session_state["detection_history"] = []
        st.session_state["detected_events"] = set()
