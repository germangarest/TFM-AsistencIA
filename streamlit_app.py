import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import time
import tempfile
import os
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av

# ------------------------- Configuración de la página -------------------------
st.set_page_config(
    page_title="Sistema de Detección de Incidentes",
    page_icon="🎥",
    layout="wide"
)

# ------------------------- Constantes y Configuración -------------------------
ALERT_THRESHOLD = 0.5  # Umbral para considerar una detección
CONFIDENCE_THRESHOLD = 0.7  # (Si se requiere en ajustes futuros)
CONSECUTIVE_FRAMES_NEEDED = 5  # Frames consecutivos necesarios para confirmar
ALERT_DURATION = 3  # Duración de la alerta visual en segundos

# Configuración para cada modelo
FIGHT_MODEL_CONFIG = {
    'img_size': 128,
    'preprocessing': lambda x: (x.astype('float32') / 255.0) - 0.5  # Normalización centrada
}

ACCIDENT_MODEL_CONFIG = {
    'img_size': 160,
    'preprocessing': lambda x: (x.astype('float32') / 255.0) - 0.5  # Normalización centrada
}

# ------------------------- Inicialización del estado de la sesión -------------------------
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
if 'detected_events' not in st.session_state:
    st.session_state.detected_events = set()
if 'current_video_name' not in st.session_state:
    st.session_state.current_video_name = None
if 'consecutive_alerts_fight' not in st.session_state:
    st.session_state.consecutive_alerts_fight = 0
if 'consecutive_alerts_accident' not in st.session_state:
    st.session_state.consecutive_alerts_accident = 0
if 'alert_active_fight' not in st.session_state:
    st.session_state.alert_active_fight = False
if 'alert_active_accident' not in st.session_state:
    st.session_state.alert_active_accident = False
if 'last_alert_time_fight' not in st.session_state:
    st.session_state.last_alert_time_fight = 0
if 'last_alert_time_accident' not in st.session_state:
    st.session_state.last_alert_time_accident = 0

# ------------------------- Cargar Modelos -------------------------
@st.cache_resource
def load_models():
    """
    Carga los modelos pre-entrenados desde la carpeta 'models'.
    Si usas una versión de Streamlit anterior a 1.18, considera usar:
      @st.cache(allow_output_mutation=True)
    """
    fight_model = tf.keras.models.load_model('models/model_fight.h5', compile=False)
    accident_model = tf.keras.models.load_model('models/model_car.h5', compile=False)
    return fight_model, accident_model

try:
    fight_model, accident_model = load_models()
    st.write("Modelos cargados exitosamente")
except Exception as e:
    st.error(f"Error al cargar los modelos: {e}")
    st.stop()

# ------------------------- Función para procesar cada frame -------------------------
def process_frame(frame):
    """
    Procesa cada frame para detectar peleas y accidentes.
    
    Args:
        frame: Imagen en BGR (numpy array).
        
    Returns:
        frame_display: Frame con las anotaciones.
        fight_prediction: Probabilidad de pelea.
        accident_prediction: Probabilidad de accidente.
    """
    # Copia para anotaciones
    frame_display = frame.copy()
    
    # --- Predicción para modelo de peleas ---
    fight_frame = cv2.resize(frame, (FIGHT_MODEL_CONFIG['img_size'], FIGHT_MODEL_CONFIG['img_size']))
    fight_frame = FIGHT_MODEL_CONFIG['preprocessing'](fight_frame)
    fight_frame = np.expand_dims(fight_frame, axis=0)
    fight_pred = float(fight_model.predict(fight_frame, verbose=0)[0][0])
    # La predicción 0 significa pelea, 1 significa no pelea
    fight_prediction = 1 - fight_pred  # Invertir el valor
    
    # --- Predicción para modelo de accidentes ---
    accident_frame = cv2.resize(frame, (ACCIDENT_MODEL_CONFIG['img_size'], ACCIDENT_MODEL_CONFIG['img_size']))
    accident_frame = ACCIDENT_MODEL_CONFIG['preprocessing'](accident_frame)
    accident_frame = np.expand_dims(accident_frame, axis=0)
    accident_pred = float(accident_model.predict(accident_frame, verbose=0)[0][0])
    # La predicción 0 significa accidente, 1 significa no accidente
    accident_prediction = 1 - accident_pred  # Invertir el valor
    
    # Asegurar valores en [0,1]
    fight_prediction = np.clip(fight_prediction, 0, 1)
    accident_prediction = np.clip(accident_prediction, 0, 1)
    
    # Añadir texto con las probabilidades y su interpretación
    cv2.putText(frame_display, f"Pelea: {fight_prediction:.1%} {'(Detectada)' if fight_prediction > ALERT_THRESHOLD else ''}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame_display, f"Accidente: {accident_prediction:.1%} {'(Detectado)' if accident_prediction > ALERT_THRESHOLD else ''}", 
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # IDs para los eventos (utilizando el nombre del video)
    fight_event_id = f"{st.session_state.current_video_name}_fight"
    accident_event_id = f"{st.session_state.current_video_name}_accident"
    
    # --- Detección de peleas ---
    if fight_prediction > ALERT_THRESHOLD:
        st.session_state.consecutive_alerts_fight += 1
        if st.session_state.consecutive_alerts_fight >= CONSECUTIVE_FRAMES_NEEDED:
            if not st.session_state.alert_active_fight:
                st.session_state.alert_active_fight = True
                st.session_state.last_alert_time_fight = time.time()
                if fight_event_id not in st.session_state.detected_events:
                    st.session_state.detected_events.add(fight_event_id)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.detection_history.append({
                        'timestamp': timestamp,
                        'type': 'Pelea',
                        'confidence': f"{fight_prediction:.1%}",
                        'video': st.session_state.current_video_name
                    })
            # Dibujar rectángulo rojo
            cv2.rectangle(frame_display, (0, 0), (frame_display.shape[1], frame_display.shape[0]), (0, 0, 255), 3)
    else:
        st.session_state.consecutive_alerts_fight = max(0, st.session_state.consecutive_alerts_fight - 1)
        if time.time() - st.session_state.last_alert_time_fight > ALERT_DURATION:
            st.session_state.alert_active_fight = False
    
    # --- Detección de accidentes ---
    if accident_prediction > ALERT_THRESHOLD:
        st.session_state.consecutive_alerts_accident += 1
        if st.session_state.consecutive_alerts_accident >= CONSECUTIVE_FRAMES_NEEDED:
            if not st.session_state.alert_active_accident:
                st.session_state.alert_active_accident = True
                st.session_state.last_alert_time_accident = time.time()
                if accident_event_id not in st.session_state.detected_events:
                    st.session_state.detected_events.add(accident_event_id)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.detection_history.append({
                        'timestamp': timestamp,
                        'type': 'Accidente',
                        'confidence': f"{accident_prediction:.1%}",
                        'video': st.session_state.current_video_name
                    })
            # Dibujar rectángulo azul
            cv2.rectangle(frame_display, (0, 0), (frame_display.shape[1], frame_display.shape[0]), (255, 0, 0), 3)
    else:
        st.session_state.consecutive_alerts_accident = max(0, st.session_state.consecutive_alerts_accident - 1)
        if time.time() - st.session_state.last_alert_time_accident > ALERT_DURATION:
            st.session_state.alert_active_accident = False
    
    return frame_display, fight_prediction, accident_prediction

# ------------------------- Clase para procesamiento en tiempo real -------------------------
class VideoProcessor:
    def __init__(self):
        self.frame_count = 0

    def recv(self, frame):
        # Convertir frame a numpy array (formato BGR)
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        processed_frame, _, _ = process_frame(img)
        # Convertir de vuelta a VideoFrame
        return av.VideoFrame.from_ndarray(processed_frame, format="bgr24")

# ------------------------- Configuración de WebRTC -------------------------
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]}
    ]}
)

# ------------------------- Interfaz principal -------------------------
st.title("Sistema de Detección de Incidentes")
st.write("""
    Este sistema analiza video en tiempo real para detectar peleas y accidentes.
    Puede utilizar su webcam o subir un video para el análisis.
""")

# Pestañas de la aplicación
tab1, tab2, tab3 = st.tabs(["Webcam", "Analizar Video", "Historial"])

# Función para crear un contenedor centralizado para video
def create_video_container():
    container = st.container()
    with container:
        cols = st.columns([1, 2, 1])
        return cols[1]

# ---------- Pestaña 1: Webcam ----------
with tab1:
    st.header("Webcam")
    
    col_video, col_metrics = st.columns([2, 1])
    
    with col_video:
        video_display = create_video_container()
        with video_display:
            webrtc_ctx = webrtc_streamer(
                key="incident-detection",
                video_processor_factory=VideoProcessor,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True
            )
            if webrtc_ctx.state.playing:
                st.session_state.current_video_name = "Webcam"
                st.session_state.detected_events = set()
            else:
                st.session_state.current_video_name = None
                st.session_state.detected_events = set()

    with col_metrics:
        st.markdown("### Métricas en Tiempo Real")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Probabilidad de Pelea", "0.0%")
            st.metric("Alertas Consecutivas", f"{st.session_state.consecutive_alerts_fight}")
        with col2:
            st.metric("Probabilidad de Accidente", "0.0%")
            st.metric("Alertas Consecutivas", f"{st.session_state.consecutive_alerts_accident}")

# ---------- Pestaña 2: Analizar Video ----------
with tab2:
    st.header("Analizar Video")
    
    video_file = st.file_uploader("Subir video", type=['mp4', 'avi', 'mov'])
    
    if video_file is not None:
        st.session_state.current_video_name = video_file.name
        st.session_state.detected_events = set()
        
        # Guardar archivo temporalmente
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        
        video_display = create_video_container()
        stframe = video_display.empty()
        
        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30  # Valor por defecto
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Probabilidades")
            prob_fight = st.empty()
            prob_accident = st.empty()
        with col2:
            st.markdown("### Estado de Alertas")
            alert_fight = st.empty()
            alert_accident = st.empty()
        
        frame_number = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame, fight_prediction, accident_prediction = process_frame(frame)
            frame_number += 1
            
            stframe.image(processed_frame, channels="BGR", use_container_width=True)
            prob_fight.metric("Probabilidad de Pelea", f"{fight_prediction:.1%}")
            prob_accident.metric("Probabilidad de Accidente", f"{accident_prediction:.1%}")
            alert_fight.metric("Alertas Consecutivas de Pelea", f"{st.session_state.consecutive_alerts_fight}")
            alert_accident.metric("Alertas Consecutivas de Accidente", f"{st.session_state.consecutive_alerts_accident}")
            
            time.sleep(0.01)
        
        cap.release()
        os.unlink(tfile.name)
        
        st.success(f"Análisis del video '{video_file.name}' completado")
        if st.session_state.detected_events:
            st.warning("Se detectaron incidentes en el video. Consulte la pestaña de Historial para más detalles.")

# ---------- Pestaña 3: Historial ----------
with tab3:
    st.header("Historial de Detecciones")
    
    if st.session_state.detection_history:
        df = pd.DataFrame(st.session_state.detection_history)
        df = df.sort_values(by=['video', 'timestamp'])
        
        st.subheader("Resumen por Video")
        video_summary = df.groupby('video').agg({
            'type': lambda x: ', '.join(sorted(set(x))),
            'confidence': 'max'
        }).reset_index()
        
        video_summary.columns = ['Video', 'Eventos Detectados', 'Confianza Máxima']
        st.dataframe(video_summary, hide_index=True)
        
        st.subheader("Detalle de Detecciones")
        st.dataframe(df[['timestamp', 'video', 'type', 'confidence']].rename(columns={
            'timestamp': 'Hora',
            'video': 'Video',
            'type': 'Tipo de Evento',
            'confidence': 'Confianza'
        }), hide_index=True)
        
        if st.button("Limpiar Historial"):
            st.session_state.detection_history = []
            st.session_state.detected_events = set()
            st.experimental_rerun()
    else:
        st.info("No hay detecciones registradas aún.")