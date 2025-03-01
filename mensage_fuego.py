import cv2
import smtplib
import numpy as np
import streamlit as st
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from tensorflow.keras.models import load_model
#from playsound import playsound
from PIL import Image

# Cargar el modelo entrenado previamente
model = load_model("./models/model_car.h5")

# Función para preprocesar cada frame antes de pasarlo al modelo
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV usa BGR, lo convertimos a RGB
    img = cv2.resize(frame, (224, 224))  # Redimensionamos al tamaño que espera el modelo
    img = img.astype("float32") / 255.0  # Normalizamos los valores a un rango entre 0 y 1
    img = np.expand_dims(img, axis=0)  # Agregamos una dimensión extra para que el modelo lo acepte
    return img

# Función para enviar un correo de alerta
def send_email_alert():
    sender_email = "www.jaradavid@gmail.com"
    receiver_email = "www.jaradavid@gmail.com"
    password = "wlspfukvtrwdkuwf"

    # Crear el mensaje con codificación UTF-8
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = '🚨 Alerta de Incendio'

    # Cuerpo del mensaje
    body = """\
    ¡Atención! Se ha detectado fuego en la cámara. Revisa la ubicación inmediatamente.
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

# Interfaz de usuario en Streamlit
st.title("Detección de Incendios en Tiempo Real")

# Subir un video desde tu máquina
video_file = st.file_uploader("Sube un video", type=["mp4", "avi", "mov"])
if video_file is not None:
    # Guardar el archivo de video temporalmente
    video_path = "/tmp/uploaded_video.mp4"
    with open(video_path, "wb") as f:
        f.write(video_file.getbuffer())

    # Iniciar la captura de video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("Error: No se pudo abrir el archivo de video.")
    else:
        alert_sent = False  # Para evitar múltiples alertas seguidas

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Fin del video

            # Preprocesamos el frame antes de pasarlo al modelo
            img = preprocess_frame(frame)

            # Hacemos la predicción con el modelo
            prediction = model.predict(img)[0][0]  # Obtener la probabilidad de fuego

            # Definir mensaje y color de alerta
            label = "NO FIRE"
            color = (0, 255, 0)  # Verde por defecto

            # Detectar fuego
            fire_detected = prediction > 0.5
            fire_pred = prediction

            if fire_detected:
                label = f"🔥 FIRE ({fire_pred:.2%})"
                color = (0, 0, 255)  # Rojo

                # Si se detecta fuego y no se ha enviado la alerta aún
                if not alert_sent:
                    st.warning("🚨 ¡ALERTA! Se ha detectado fuego.")
                    #playsound("alarma.mp3")  # Sonido de alerta
                    send_email_alert()  # Enviar email
                    alert_sent = True  # Evitar alertas repetidas

            # Dibujar la etiqueta sobre el frame
            cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Mostrar el frame procesado en Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convertir a RGB para mostrarlo con PIL
            image = Image.fromarray(frame_rgb)
            st.image(image, channels="RGB", caption="Frame de Video", use_column_width=True)

            # Hacer una pausa para simular un video en tiempo real (puedes ajustar esto)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()

else:
    st.info("Por favor sube un video para comenzar la detección.")
