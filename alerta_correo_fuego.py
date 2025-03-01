import smtplib
import streamlit as st
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

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

# Parámetros de ejemplo
fire_detected = True
fire_pred = 0.9

# Para evitar múltiples alertas seguidas
alert_sent = False
if fire_detected:
    label = f"🔥 FIRE ({fire_pred:.2%})"
    color = (0, 0, 255)  # Rojo

    # Si se detecta fuego y no se ha enviado la alerta aún
    if not alert_sent:
        st.warning("🚨 ¡ALERTA! Se ha detectado fuego.")
        #playsound("alarma.mp3")  # Sonido de alerta
        send_email_alert()  # Enviar email
        alert_sent = True  # Evitar alertas repetidas