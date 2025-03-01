# streamlit_app.py
import streamlit as st
import os

# Configuración de la página principal
st.set_page_config(
    page_title="AsistencIA - Sistema de Detección de Incidentes", 
    page_icon="🚨", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Función para cargar CSS externo
def load_css(css_file):
    if os.path.exists(css_file):
        with open(css_file, 'r', encoding='utf-8') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        # Si no encuentra el archivo, crea la carpeta
        os.makedirs(os.path.dirname(css_file), exist_ok=True)
        st.warning(f"No se encontró el archivo CSS: {css_file}")

# Intentar cargar el archivo CSS externo
load_css('static/styles.css')

# --- Control de pestañas ---
# Al iniciar la aplicación se define que la pestaña actual es "Detección"
if "current_tab" not in st.session_state:
    st.session_state.current_tab = "Detección"

# Mejorar la apariencia de la barra lateral
with st.sidebar:
    # Logo y título con mejor diseño
    st.markdown(
        '''
        <div style="text-align:center; padding: 15px 0; margin-bottom:25px;">
            <div style="font-size:3rem; margin-bottom:10px;">🚨</div>
            <h2 style="margin:0; font-size:1.8rem; font-weight:700;">AsistencIA</h2>
            <p style="margin:5px 0 0 0; opacity:0.8; font-size:0.9rem;">Sistema de detección de incidentes</p>
        </div>
        ''', 
        unsafe_allow_html=True
    )
    
    # Navegación con botones mejorados
    st.markdown("### Navegación")
    
    # Contenedor para los botones de navegación
    col1, col2 = st.columns(2)
    
    if col1.button("📹 Detección", 
                  key="btn_detection",
                  use_container_width=True,
                  type="primary" if st.session_state.current_tab == "Detección" else "secondary"):
        st.session_state.current_tab = "Detección"
        if 'run' in st.session_state:
            st.session_state['run'] = False
        st.rerun()
    
    if col2.button("💬 Chatbot", 
                  key="btn_chatbot",
                  use_container_width=True,
                  type="primary" if st.session_state.current_tab == "Chatbot" else "secondary"):
        st.session_state.current_tab = "Chatbot"
        st.rerun()
    
    # Agregar el nuevo botón para el Agente
    if st.button("🧠 Agente", 
               key="btn_agent",
               use_container_width=True,
               type="primary" if st.session_state.current_tab == "Agente" else "secondary"):
        st.session_state.current_tab = "Agente"
        st.rerun()

# Luego actualizar las condiciones para ejecutar cada pestaña
if st.session_state.current_tab == "Detección":
    exec(open("deteccion.py", encoding="utf-8").read())
elif st.session_state.current_tab == "Chatbot":
    exec(open("chatbot.py", encoding="utf-8").read())
elif st.session_state.current_tab == "Agente":
    exec(open("agente.py", encoding="utf-8").read())