import streamlit as st
import cv2
import pandas as pd
import numpy as np
import tempfile
import av
import os
from datetime import datetime
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
import torch
from typing import Dict, List

# ===============================
# CSS personalizado
# ===============================
st.markdown("""
<style>
    /* Contenedor principal */
    .main .block-container {
        max-width: 95%;
        padding-top: 2rem;
    }
    /* Video en tiempo real */
    .stVideo {
        border-radius: 10px;
        margin: 20px 0;
        width: 90%;
        max-width: 1200px;
    }
    /* Historial expandido */
    .historial-item {
        width: 100%;
        margin: 10px 0;
        padding: 15px;
        font-size: 1.1rem;
    }
    /* Títulos más grandes */
    h1 {
        font-size: 3rem !important;
        margin-bottom: 30px !important;
    }
    .header {
        color: #1a73e8;
        border-bottom: 2px solid #1a73e8;
        padding-bottom: 10px;
    }
    .alert-box {
        background-color: #ffcccc;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        border: 1px solid #ff4444;
    }
    .historial-item {
        padding: 10px;
        margin: 5px 0;
        background-color: #f8f9fa;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .header-box {
        background-color: #e8f0fe;
        padding: 15px;
        border-left: 5px solid #1a73e8;
        margin-bottom: 20px;
        border-radius: 5px;
        display: inline-block;
    }

    @media (prefers-color-scheme: dark) {
        .header-box {
            background-color: #e8f0fe;  /* Fondo más oscuro */
            border-left: 5px solid #4a90e2;  /* Color que contraste bien */
        }
    }
    .report-font {
        font-family: Arial, sans-serif;
        font-size: 0.9rem;
    }
    div[data-baseweb="radio"] > label {
        font-size: 0.85rem;
        margin-right: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ===============================
# Cargar modelo unificado usando UnifiedModel
# ===============================
from unified_model import UnifiedModel

@st.cache_resource
def load_unified_model():
    return UnifiedModel(device="cpu")

# ===============================
# Función de dibujo optimizada
# ===============================
def draw_detections(frame: np.ndarray, detections: Dict[int, List], classes: Dict) -> np.ndarray:
    for class_id in detections:
        label, color = classes[class_id]
        for box in detections[class_id]:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {box[4]:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

# ===============================
# Función para actualizar historial
# ===============================
def update_history(detections: Dict[int, List], source: str):
    class_mapping = {0: "Accidente", 1: "Pelea", 2: "Incendio"}
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    for class_id in detections:
        for box in detections[class_id]:
            # Corregir la verificación de longitud del box
            if len(box) < 5:  # Cambiado de 6 a 5
                continue
            
            conf = box[4]
            if conf >= 0.5:
                entry = {
                    'Hora de la detección': timestamp,
                    'Fuente': source,
                    'Tipo de incidente': class_mapping[class_id],
                    'Precisión': f"{conf:.2f}",
                }
                st.session_state.history.append(entry)

# ===============================
# Optimización del sistema
# ===============================
import psutil

def optimize_system():
    process = psutil.Process()
    if os.name == "nt": 
        process.nice(psutil.HIGH_PRIORITY_CLASS)
    else: 
        process.nice(-10)
    cv2.setNumThreads(8)
    cv2.ocl.setUseOpenCL(True)

optimize_system()

# ===============================
# Procesador de video optimizado para la webcam
# ===============================
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = load_unified_model()
        self.frame_skip = 5
        self.frame_count = 0
        self.last_detections = None  # Almacena las detecciones del último frame procesado
        self.classes = {
            0: ("Accidente", (0, 0, 255)),
            1: ("Pelea", (255, 0, 0)),
            2: ("Incendio", (0, 255, 0))
        }

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        original_h, original_w = img.shape[:2]
        
        # Redimensionar para detección
        detection_frame = cv2.resize(img, (640, 480))
        
        if self.frame_count % self.frame_skip == 0:
            img_rgb = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
            detections = self.model.detect(img_rgb)
            
            # Escalar coordenadas correctamente
            scale_x = original_w / 640
            scale_y = original_h / 480
            scaled_detections = {}
            for class_id in detections:
                scaled_boxes = []
                for box in detections[class_id]:
                    x1, y1, x2, y2, conf = box[:5]
                    scaled_boxes.append([
                        x1 * scale_x,
                        y1 * scale_y,
                        x2 * scale_x,
                        y2 * scale_y,
                        conf
                    ])
                scaled_detections[class_id] = scaled_boxes
            
            self.last_detections = scaled_detections  # Almacenar detecciones actuales
            update_history(scaled_detections, "Webcam")
            img = draw_detections(img, scaled_detections, self.classes)
        else:
            # En frames intermedios, si existen detecciones previas, se dibujan sobre el frame
            if self.last_detections is not None:
                img = draw_detections(img, self.last_detections, self.classes)
        
        self.frame_count += 1
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ===============================
# Interfaz principal
# ===============================
def main():
    st.title("🚨 AsistencIA - sistema de detección de incidentes")
    # Sidebar de la aplicación
    with st.sidebar:
        # Información sobre la aplicación
        st.markdown("---")
        st.markdown("### Descripción")
        st.write("En este apartado se podrán detectar tres tipos de incidentes para que los servicios de emergencia puedan ofrecer una temprana ayuda.")
        st.write("Hecho con ❤️ por los estudiantes de **Accenture**")
        st.markdown("---")

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

    active_tab = st.radio("Selecciona una opción:", 
                         ["Webcam", "Analizar Video", "Historial"], 
                         horizontal=True, key="active_tab")
    
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # PESTAÑA: WEBCAM
    if active_tab == "Webcam":
        activar_cam = st.checkbox("Activar cámara", key="camera_activation")
        
        if activar_cam:
            webrtc_ctx = webrtc_streamer(
                key="asistencia",
                video_processor_factory=VideoProcessor,
                async_processing=False,  # Procesamiento síncrono para mejor debug
                media_stream_constraints={
                    "video": {"width": 1280, "height": 720, "frameRate": 20},
                    "audio": False
                },
                video_html_attrs={
                    "style": {
                    "width": "100%", 
                    "maxWidth": "800px"
                    }
                },
                desired_playing_state=activar_cam
            )

    # PESTAÑA: ANALIZAR VIDEO
    elif active_tab == "Analizar Video":
        uploaded_file = st.file_uploader("Subir video", type=["mp4", "avi", "mov"])
        if uploaded_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
        
            stframe = st.empty()
            cap = cv2.VideoCapture(tfile.name)
            model = load_unified_model()
            classes = {
                0: ("Accidente", (0, 0, 255)),
                1: ("Pelea", (255, 0, 0)),
                2: ("Incendio", (0, 255, 0))
            }
        
            # Configuración del frame skip
            frame_skip = 2 
            frame_count = 0
            last_detections = None  # Variable para almacenar las detecciones del último frame procesado

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                frame_count += 1

                # Redimensionamos el frame para mantener consistencia en el dibujo
                frame = cv2.resize(frame, (640, 480))
            
                # Verifica si el frame actual debe procesarse
                if frame_count % frame_skip != 0:
                    # Si hay detecciones previas, las dibuja sobre el frame actual
                    if last_detections is not None:
                        frame = draw_detections(frame, last_detections, classes)
                    stframe.image(frame, channels="BGR", width=800)
                    continue
            
                # Procesa el frame: convierte a RGB y detecta
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = model.detect(frame_rgb)
                last_detections = detections  # Almacena las detecciones actuales para usarlas en frames intermedios
            
                # Dibuja las detecciones en el frame
                frame = draw_detections(frame, detections, classes)
                stframe.image(frame, channels="BGR", width=800)
                update_history(detections, uploaded_file.name)
            
            cap.release()
            os.unlink(tfile.name)

    # PESTAÑA: HISTORIAL
    else:
        st.subheader("Registro de incidentes")
        
        if not st.session_state.history:
            st.info("No se han registrado incidentes")
        else:
            # Crear DataFrame con los datos
            df = pd.DataFrame(st.session_state.history)
            
            # Mostrar tabla con los datos
            st.dataframe(df)
            
            # Crear botón de descarga
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descargar CSV completo",
                data=csv,
                file_name="historial_incidentes.csv",
                mime="text/csv",
                key="download-csv"
            )

        if st.button("Limpiar historial", use_container_width=True):
            st.session_state.history = []

if __name__ == "__main__":
    main()