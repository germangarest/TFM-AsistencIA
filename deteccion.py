# deteccion.py
import streamlit as st
import cv2
import pandas as pd
import numpy as np
import tempfile
import av
import os
from datetime import datetime
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
import threading
import queue
import time
from typing import Dict, List
import psutil

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ===============================
# Función para enviar un correo de alerta
# ===============================
# Función para enviar un correo de alerta
def send_email_alert(emergencia):
    sender_email = "www.jaradavid@gmail.com"
    receiver_email = "www.jaradavid@gmail.com"
    password = "wlspfukvtrwdkuwf"

    # Crear el mensaje con codificación UTF-8
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = '🚨 Alerta de emergencia'

    # Cuerpo del mensaje
    body = """\
    ¡Atención! Se ha detectado una emergencia en la cámara. Revisa la ubicación inmediatamente.
    """
    msg.attach(MIMEText(body, 'plain', 'utf-8'))

    try:
        # Conexión al servidor SMTP de Gmail
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()  # Iniciar la conexión segura
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        st.success("✅ Alerta de email enviada.")
    except Exception as e:
        st.error(f"⚠️ No se pudo enviar el email: {e}")

# Global variable para el umbral de confianza
CONF_THRESHOLD = 0.45

# ===============================
# Precarga del modelo unificado
# ===============================
from unified_model import UnifiedModel
@st.cache_resource
def load_unified_model():
    model = UnifiedModel(device="cpu")
    dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    model.detect(dummy)
    return model

if 'model_loaded' not in st.session_state:
    load_unified_model()
    st.session_state.model_loaded = True

# ===============================
# Función de dibujo optimizada
# ===============================
def draw_detections(frame, detections, classes):
    for class_id in detections:
        label, color = classes[class_id]
        for box in detections[class_id]:
            x1, y1, x2, y2 = map(int, box[:4])
            conf = box[4]
            if conf >= CONF_THRESHOLD:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text = f"{label} {conf:.0%}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (x1, y1 - 25), (x1 + tw, y1), color, -1)
                cv2.putText(frame, text, (x1 + 2, y1 - 7), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    return frame

# ===============================
# Función para actualizar historial
# ===============================
def update_history(detections: Dict[int, List], source: str):
    class_mapping = {0: "Accidente", 1: "Pelea", 2: "Incendio"}
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entries = []
    for class_id in detections:
        for box in detections[class_id]:
            if len(box) < 5:
                continue
            conf = box[4]
            if conf >= CONF_THRESHOLD:
                entry = {
                    'Hora de la detección': timestamp,
                    'Fuente': source,
                    'Tipo de incidente': class_mapping.get(class_id, "Desconocido"),
                    'Precisión': f"{conf:.2f}",
                }
                entries.append(entry)
    if entries:
        with threading.Lock():
            if 'history' in st.session_state:
                st.session_state.history.extend(entries)
            else:
                st.session_state.history = entries

# ===============================
# Optimización del sistema
# ===============================
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
# Procesador de video optimizado
# ===============================
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self._model = None
        self.detection_queue = queue.Queue(maxsize=1)
        self.classes = {
            0: ("Accidente", (255, 0, 0)),     # Azul
            1: ("Pelea", (0, 165, 255)),       # Naranja
            2: ("Incendio", (0, 0, 255))       # Rojo
        }
        self.processing = False
        self.last_detections = None
        self.target_size = 640
        self.conf_threshold = CONF_THRESHOLD
        self.latest_frame = None
        threading.Thread(target=self.process_frames, daemon=True).start()
    
    @property
    def model(self):
        if self._model is None:
            self._model = UnifiedModel(device="cpu")
            self._model.detect(np.zeros((640, 640, 3), dtype=np.uint8))
        return self._model
    
    def process_frames(self):
        while True:
            if self.latest_frame is not None and not self.processing:
                self.processing = True
                try:
                    h, w = self.latest_frame.shape[:2]
                    img_resized, ratio, pad = self.letterbox(self.latest_frame)
                    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                    raw_detections = self.model.detect(img_rgb)
                    processed_detections = self.scale_coords(raw_detections, ratio, pad, (h, w))
                    self.last_detections = processed_detections
                except Exception as e:
                    print(f"Error crítico en process_frames: {str(e)}")
                    time.sleep(1)
                finally:
                    self.processing = False
                    self.latest_frame = None
                time.sleep(0.03)
            else:
                time.sleep(0.005)
    
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.latest_frame = img.copy()
        if self.last_detections:
            img = self.draw_realtime_detections(img, self.last_detections)
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    def letterbox(self, img):
        shape = img.shape[:2]
        new_shape = (self.target_size, self.target_size)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw = new_shape[1] - new_unpad[0]
        dh = new_shape[0] - new_unpad[1]
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)
        dw /= 2
        dh /= 2
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return img, r, (dw, dh)
    
    def scale_coords(self, detections, ratio, pad, original_shape):
        processed = {0: [], 1: [], 2: []}
        dw, dh = pad
        for class_id in detections:
            for box in detections[class_id]:
                x1, y1, x2, y2, conf = box[:5]
                x1 = np.clip((x1 - dw) / ratio, 0, original_shape[1])
                y1 = np.clip((y1 - dh) / ratio, 0, original_shape[0])
                x2 = np.clip((x2 - dw) / ratio, 0, original_shape[1])
                y2 = np.clip((y2 - dh) / ratio, 0, original_shape[0])
                processed[class_id].append([
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2),
                    conf
                ])
        return processed
    
    def draw_realtime_detections(self, frame, detections):
        for class_id in detections:
            label, color = self.classes[class_id]
            for box in detections[class_id]:
                x1, y1, x2, y2 = box[:4]
                conf = box[4]
                if conf >= self.conf_threshold:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    text = f"{label} {conf:.0%}"
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(frame, (x1, y1 - 25), (x1 + tw, y1), color, -1)
                    cv2.putText(frame, text, (x1 + 2, y1 - 7), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        return frame

# ===============================
# Configuración del streamer con sincronización de rendimiento
# ===============================
def video_streamer():
    return webrtc_streamer(
        key="asistencia",
        video_processor_factory=VideoProcessor,
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 10},
            },
            "audio": False
        },
        video_html_attrs={
            "style": {"width": "100%", "maxWidth": "900px", "borderRadius": "8px", "backgroundColor": "#121212", "border": "none", "boxShadow": "none"},
            "autoPlay": True,
            "playsInline": True
        },
        desired_playing_state=True,
        sendback_audio=False
    )

# ===============================
# Interfaz principal mejorada
# ===============================
def main():
    st.title("🚨 Sistema de detección de incidentes")
    
    # Información del proyecto en un contenedor destacado
    st.markdown(
        """
        <div class="header-box">
            <p>Sistema inteligente para la detección automática de situaciones de emergencia mediante cámaras de vigilancia.</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Uso de pestañas para mejor organización y experiencia de usuario
    tab1, tab2, tab3 = st.tabs([
        "📹 Webcam", 
        "🎥 Analizar video", 
        "📊 Historial de incidentes"
    ])
    
    # Inicializar historial si no existe
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # Pestaña 1: Detección en tiempo real
    with tab1:
        col1, col2 = st.columns([7, 3])
        
        with col1:
            st.subheader("Detección mediante cámara")
            st.write("Utiliza la cámara de tu dispositivo para detectar incidentes en tiempo real.")
            
            activar_cam = st.toggle("Activar cámara", key="camera_activation")
            
            if activar_cam:
                video_streamer()
            else:
                # Mostrar imagen placeholder cuando la cámara está desactivada
                st.markdown(
                    """
                    <div class="camera-placeholder">
                        <div style="font-size:3rem; margin-bottom:10px;">📹</div>
                        <p>Activa la cámara para comenzar la detección</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        
        with col2:
            st.markdown("### Incidentes detectables")
            
            # Tarjetas informativas para cada tipo de incidente
            st.markdown(
                """
                <div class="incident-card accident">
                    <h4 style="margin:0 0 5px 0;">🚗 Accidentes de coche</h4>
                    <p style="margin:0; font-size:0.9rem;">Colisiones de vehículos.</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            st.markdown(
                """
                <div class="incident-card fight">
                    <h4 style="margin:0 0 5px 0;">👥 Peleas</h4>
                    <p style="margin:0; font-size:0.9rem;">Enfrentamientos físicos entre personas.</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            st.markdown(
                """
                <div class="incident-card fire">
                    <h4 style="margin:0 0 5px 0;">🔥 Incendios</h4>
                    <p style="margin:0; font-size:0.9rem;">Fuego y situaciones de combustión peligrosa.</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
    
    # Pestaña 2: Análisis de video
    with tab2:
        st.subheader("Análisis de video grabado")
        st.write("Sube un video para que el sistema detecte incidentes automáticamente.")
        
        # Inicializar variables de estado si no existen
        if "video_uploaded" not in st.session_state:
            st.session_state.video_uploaded = False
            st.session_state.analysis_complete = False
            st.session_state.video_file = None
            st.session_state.video_stats = {"Accidente": 0, "Pelea": 0, "Incendio": 0, "Total": 0}
            st.session_state.last_frame = None  # Para guardar el último frame
        
        # Contenedor para el uploader que podemos ocultar/mostrar
        uploader_container = st.empty()
        
        # Diseño principal de dos columnas
        col1, col2 = st.columns([7, 3])
        
        with col1:
            # Solo mostrar el uploader si no hay video cargado
            if not st.session_state.video_uploaded:
                with uploader_container:
                    uploaded_file = st.file_uploader(
                        "Selecciona un archivo de video para analizar", 
                        type=["mp4", "avi", "mov", "mkv"]
                    )
                    
                    if uploaded_file:
                        # Guardar el archivo en session_state
                        st.session_state.video_file = uploaded_file
                        # Marcar que tenemos un video cargado
                        st.session_state.video_uploaded = True
                        st.session_state.analysis_complete = False
                        # Ocultar el uploader
                        uploader_container.empty()
                        # Recargar para aplicar los cambios en la UI
                        st.experimental_rerun()
            
            # Contenedor para el video (siempre visible)
            if st.session_state.video_uploaded:
                # Contenedor para el video
                video_container = st.container()
                
                # Mostrar el video, ya sea en análisis o completado
                with video_container:
                    # Usar el archivo guardado en session_state
                    uploaded_file = st.session_state.video_file
                    
                    # Información del archivo en expandible
                    with st.expander("Detalles del archivo", expanded=False):
                        file_details = {
                            "Nombre": uploaded_file.name, 
                            "Tipo": uploaded_file.type, 
                            "Tamaño": f"{uploaded_file.size / 1024 / 1024:.2f} MB"
                        }
                        for key, value in file_details.items():
                            st.markdown(f"**{key}:** {value}")
                    
                    # Crear contenedor para video con tamaño controlado y margen inferior reducido
                    st.markdown(
                        """
                        <div class="bordered-container" style="padding: 0; overflow: hidden; border-radius: 8px; margin-bottom: 10px;">
                            <h4 style="background-color: #2a2a2a; margin: 0; padding: 10px 15px; border-bottom: 1px solid #444;">
                                📹 Video analizado
                            </h4>
                            <div style="padding: 15px; display: flex; justify-content: center;">
                        """, 
                        unsafe_allow_html=True
                    )
                    
                    # Contenedor para el frame actual
                    frame_container = st.empty()
                    
                    # Si el análisis ya está completo, mostrar el último frame
                    if st.session_state.analysis_complete and st.session_state.last_frame is not None:
                        frame_container.image(
                            st.session_state.last_frame, 
                            channels="BGR", 
                            width=640  # Ancho cambiado a 640
                        )
                        st.markdown("</div></div>", unsafe_allow_html=True)
                    
                    # Si el análisis no está completo, comenzar el procesamiento
                    elif not st.session_state.analysis_complete:
                        # Procesamiento del video
                        with st.spinner("Procesando video... Por favor espere."):
                            # Crear archivo temporal
                            tfile = tempfile.NamedTemporaryFile(delete=False)
                            tfile.write(uploaded_file.read())
                            tfile.close()
                            
                            # Mejorar la visualización del progreso
                            progress_text = "Analizando video"
                            progress_bar = st.progress(0, text=progress_text)
                            
                            # Configuración y procesamiento del video
                            cap = cv2.VideoCapture(tfile.name)
                            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            model = load_unified_model()
                            classes = {
                                0: ("Accidente", (255, 0, 0)),
                                1: ("Pelea", (0, 165, 255)),
                                2: ("Incendio", (0, 0, 255))
                            }
                            
                            frame_count = 0
                            last_detections = None
                            frame_skip = 8
                            vp = VideoProcessor()
                            
                            # Contadores para estadísticas
                            stats = {"Accidente": 0, "Pelea": 0, "Incendio": 0, "Total": 0}
                            
                            while cap.isOpened():
                                success, frame = cap.read()
                                if not success:
                                    break
                                
                                frame_count += 1
                                progress_value = min(frame_count / total_frames, 1.0)
                                progress_bar.progress(
                                    progress_value, 
                                    text=f"{progress_text} ({frame_count}/{total_frames} frames - {progress_value:.0%})"
                                )
                                
                                original_frame = frame.copy()
                                if frame_count % frame_skip != 0:
                                    if last_detections is not None:
                                        frame = draw_detections(original_frame, last_detections, classes)
                                    frame_container.image(
                                        frame, 
                                        channels="BGR", 
                                        width=640  
                                    )
                                    continue
                                
                                img_resized, ratio, pad = vp.letterbox(frame)
                                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                                detections = model.detect(img_rgb)
                                processed_detections = vp.scale_coords(detections, ratio, pad, frame.shape[:2])
                                last_detections = processed_detections
                                
                                # Actualizar estadísticas
                                for class_id, boxes in processed_detections.items():
                                    class_name = classes[class_id][0]
                                    for box in boxes:
                                        if len(box) >= 5 and box[4] >= CONF_THRESHOLD:
                                            stats[class_name] += 1
                                            stats["Total"] += 1
                                
                                frame = draw_detections(original_frame, processed_detections, classes)
                                frame_container.image(
                                    frame, 
                                    channels="BGR", 
                                    width=640  # Ancho cambiado a 640
                                )
                                
                                # Guardar el último frame para mostrarlo después
                                st.session_state.last_frame = frame.copy()
                                
                                update_history(processed_detections, uploaded_file.name)
                            
                            # Cerrar recursos
                            cap.release()
                            os.unlink(tfile.name)
                            
                            # Guardar estadísticas en session_state
                            st.session_state.video_stats = stats
                            
                            # Marcar que el análisis está completo
                            st.session_state.analysis_complete = True
                            
                            # Cerrar el div del contenedor de video
                            st.markdown("</div></div>", unsafe_allow_html=True)
                            
                            # Recargar para mostrar los resultados finales
                            st.experimental_rerun()
        
        with col2:
            st.markdown("### Recomendaciones")
            
            # Consejos útiles con mejor presentación
            st.markdown(
                """
                <div class="bordered-container">
                    <h4 style="margin-top:0;">Para mejores resultados:</h4>
                    <ul style="margin-bottom:0; padding-left:20px;">
                        <li><strong>Formato:</strong> preferentemente MP4</li>
                        <li><strong>Duración:</strong> videos cortos para análisis rápido</li>
                        <li><strong>Resolución:</strong> mínimo 640x480 píxeles</li>
                        <li><strong>Iluminación:</strong> adecuada para mejor detección</li>
                        <li><strong>Movimiento:</strong> evitar excesivo movimiento de cámara</li>
                    </ul>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # Mostrar botón para nuevo análisis si ya hay resultados
            if st.session_state.analysis_complete:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🔄 Analizar otro video"):
                    # Reiniciar estado
                    st.session_state.video_uploaded = False
                    st.session_state.analysis_complete = False
                    st.session_state.video_file = None
                    st.session_state.last_frame = None
                    st.experimental_rerun()
        
        # Mostrar resultados del análisis después de completado
        if st.session_state.analysis_complete:
            # Obtener estadísticas guardadas
            stats = st.session_state.video_stats
            
            # Contenedor para resultados con estilo y margen inferior reducido
            st.markdown(
                """
                <div class="bordered-container" style="padding: 0; margin-top: 10px; margin-bottom: 10px;">
                    <h3 style="background-color: #2a2a2a; margin: 0; padding: 15px; border-bottom: 1px solid #444;">
                        📊 Resultados del análisis
                    </h3>
                    <div style="padding: 20px;">
                """, 
                unsafe_allow_html=True
            )
            
            # Mostrar resultados
            if any(stats.values()):
                st.success(f"Análisis completado: se han detectado incidentes")
                
                # Mostrar las estadísticas en tarjetas atractivas
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(
                        f"""
                        <div class="incident-card accident" style="text-align: center;">
                            <h4 style="margin:0;">🚗 Accidentes</h4>
                            <p style="font-size:1.2rem; font-weight:bold; margin:10px 0;">{"✅ Detectado" if stats["Accidente"] else "❌ No detectado"}</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                
                with col2:
                    st.markdown(
                        f"""
                        <div class="incident-card fight" style="text-align: center;">
                            <h4 style="margin:0;">👥 Peleas</h4>
                            <p style="font-size:1.2rem; font-weight:bold; margin:10px 0;">{"✅ Detectado" if stats["Pelea"] else "❌ No detectado"}</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                
                with col3:
                    st.markdown(
                        f"""
                        <div class="incident-card fire" style="text-align: center;">
                            <h4 style="margin:0;">🔥 Incendios</h4>
                            <p style="font-size:1.2rem; font-weight:bold; margin:10px 0;">{"✅ Detectado" if stats["Incendio"] else "❌ No detectado"}</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    
                # Enviar correo de alerta
                if stats['Total'] > 0:
                    send_email_alert()  # Enviar email
                    alert_sent = True  # Evitar alertas repetidas                        
                
                # Se ha eliminado la gráfica de barras
            else:
                st.info("No se detectaron incidentes en este video")
            
            st.markdown("</div></div>", unsafe_allow_html=True)
    
    # Pestaña 3: Historial de incidentes
    with tab3:
        st.subheader("Registro histórico de incidentes")
        
        # Barra de acciones para gestionar el historial
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🗑️ Limpiar historial"):
                st.session_state.history = []
                st.success("Historial eliminado correctamente")
                st.experimental_rerun()
        
        with col2:
            if st.session_state.history:
                df = pd.DataFrame(st.session_state.history)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Descargar CSV",
                    data=csv,
                    file_name=f"historial_incidentes_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                )
        
        # Mostrar contenido del historial
        if not st.session_state.history:
            # Mensaje cuando no hay datos
            st.markdown(
                """
                <div class="bordered-container" style="text-align:center; padding:30px 20px;">
                    <div style="font-size:3rem; margin-bottom:10px;">📊</div>
                    <h3 style="margin:0 0 10px 0;">No hay incidentes registrados</h3>
                    <p style="margin:0;">Los incidentes detectados aparecerán aquí automáticamente</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
        else:
            # Convertir historial a DataFrame
            df = pd.DataFrame(st.session_state.history)
            
            # Preparar datos para visualizaciones
            if 'Tipo de incidente' in df.columns:
                incident_counts = df['Tipo de incidente'].value_counts().reset_index()
                incident_counts.columns = ['Tipo', 'Cantidad']
                
                # Convertir precisión a números
                if 'Precisión' in df.columns and df['Precisión'].dtype == 'object':
                    df['Precisión'] = df['Precisión'].astype(float)
                
                # Mostrar métricas resumidas
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Total de incidentes", 
                        value=len(df)
                    )
                
                with col2:
                    if not incident_counts.empty:
                        most_common_type = incident_counts.iloc[0]['Tipo']
                        most_common_count = incident_counts.iloc[0]['Cantidad']
                        st.metric(
                            label="Tipo más frecuente", 
                            value=most_common_type,
                            delta=f"{most_common_count} detectados"
                        )
                
                with col3:
                    if 'Precisión' in df.columns:
                        avg_precision = df['Precisión'].mean()
                        st.metric(
                            label="Precisión media", 
                            value=f"{avg_precision:.2f}"
                        )
                
                # Dos pestañas para visualizaciones y tabla
                viz_tab1, viz_tab2 = st.tabs(["📊 Visualizaciones", "📋 Datos detallados"])
                
                with viz_tab1:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Distribución por tipo")
                        st.bar_chart(incident_counts, x='Tipo', y='Cantidad')
                    
                    with col2:
                        if 'Hora de la detección' in df.columns:
                            st.subheader("Incidentes por día")
                            # Crear columna de fecha sin hora
                            df['Fecha'] = pd.to_datetime(df['Hora de la detección']).dt.date
                            date_counts = df.groupby('Fecha').size().reset_index(name='Cantidad')
                            date_counts = date_counts.sort_values('Fecha')
                            
                            # Generar gráfico de líneas
                            st.line_chart(date_counts, x='Fecha', y='Cantidad')
                
                with viz_tab2:
                    # Filtrar por tipo
                    if 'Tipo de incidente' in df.columns:
                        tipos = df['Tipo de incidente'].unique()
                        selected_tipos = st.multiselect(
                            "Filtrar por tipo:", 
                            options=tipos,
                            default=list(tipos)
                        )
                        
                        # Aplicar filtros
                        if selected_tipos:
                            filtered_df = df[df['Tipo de incidente'].isin(selected_tipos)]
                            # Mostrar tabla
                            st.dataframe(filtered_df, width=None)
                        else:
                            st.warning("Selecciona al menos un tipo de incidente para ver los datos")
                    else:
                        st.dataframe(df, width=None)

if __name__ == "__main__":
    main()