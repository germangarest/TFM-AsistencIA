import streamlit as st

# Configuración de la página principal
st.set_page_config(page_title="AsistencIA - sistema de deteccion de incidentes", page_icon="🚨", layout="wide")

# Añadir CSS personalizado para un menú atractivo
st.markdown(
    """
    <style>
    /* Estilo para la barra lateral */
    .css-1d391kg {
        background: linear-gradient(135deg, #0099cc, #66ccff);
        color: white;
    }
    /* Estilo para títulos y párrafos en la barra lateral */
    .sidebar .sidebar-content h1,
    .sidebar .sidebar-content h2,
    .sidebar .sidebar-content h3,
    .sidebar .sidebar-content p {
        color: white;
        font-family: 'Helvetica', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Control de pestañas ---
# Al iniciar la aplicación se define que la pestaña actual es "Detección"
if "current_tab" not in st.session_state:
    st.session_state.current_tab = "Detección"

# El menú lateral permite cambiar entre "Detección" y "Chatbot"
selected_tab = st.sidebar.radio("Navegación", ["Detección", "Chatbot"], index=0)

# Si se cambia de pestaña, se limpian los procesos de la pestaña anterior.
if st.session_state.current_tab != selected_tab:
    if st.session_state.current_tab == "Detección":
        st.session_state['run'] = False  # Esto desactiva la webcam o video en la pestaña de Detección
    # Si en el futuro se agregan procesos en Chatbot, se pueden limpiar aquí.
    st.session_state.current_tab = selected_tab

# Ejecutar la pestaña correspondiente
if selected_tab == "Detección":
    exec(open("deteccion.py", encoding="utf-8").read())
elif selected_tab == "Chatbot":
    exec(open("chatbot.py", encoding="utf-8").read())