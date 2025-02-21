import os
import pickle
import streamlit as st
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, Document
from llama_index.embeddings.deepinfra import DeepInfraEmbeddingModel
from llama_index.llms.deepinfra import DeepInfraLLM
from pypdf import PdfReader

# Configuración de la aplicación Streamlit (Mover a la parte superior)
st.set_page_config(page_title="Asistente de Emergencia", page_icon="🚨", layout="centered")

# Configuración de DeepInfra API
# Cargar variables de entorno desde .env
load_dotenv()
deepinfra_api_key = os.getenv("DEEPINFRA_TOKEN")

if not deepinfra_api_key:
    # Cargar la API Key de DeepInfra desde las secrets de Streamlit Cloud
    deepinfra_api_key = st.secrets["DEEPINFRA_TOKEN"]
    
    if not deepinfra_api_key:
        st.error("Falta la API Key de DeepInfra. Configúrala en un archivo .env o en el apartado secrets de stramlit cloud")
    

# Definir el modelo de embeddings y LLM de DeepInfra
# Configurar el modelo de embeddings con el nuevo modelo BGAI/bge-m3
Settings.embed_model = DeepInfraEmbeddingModel(
    model_id="BAAI/bge-m3",  # ID del modelo de embeddings de DeepInfra
    api_token=deepinfra_api_key,  # Token de la API
    normalize=True,  # Normalización opcional
    text_prefix="text: ",  # Prefijo de texto
    query_prefix="query: ",  # Prefijo de consulta
)

Settings.chunk_size = 1024

# Crear el modelo de embeddings
embedding_model = Settings.embed_model

# Configuración centralizada del modelo LLM utilizando Settings
Settings.llm = DeepInfraLLM(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",  # Modelo de DeepInfra
    api_key=deepinfra_api_key,  # Tu clave API de DeepInfra
    temperature=0,  # Configuración de temperatura para control de creatividad
)

# Ruta del índice guardado
INDEX_PATH = "./vector_index.pkl"

@st.cache_resource
def extract_text_from_pdf(file_path):
    """Extrae el texto completo de un PDF usando pypdf."""
    reader = PdfReader(file_path)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

# Función para cargar y guardar el índice y el modelo de embeddings
@st.cache_resource
def load_index():

    """Carga el índice de documentos. Si ya existe un índice guardado, lo reutiliza."""
    if os.path.exists(INDEX_PATH):
        # El "rb" es para la lectura binaria del documento
        with open(INDEX_PATH, "rb") as f:
            return pickle.load(f)
        
    # Los documentos están en esta carpeta
    docs_folder = "docs"
    documents = []

    # Recorre cada archivo en la carpeta "docs"
    for file_name in os.listdir(docs_folder):
        file_path = os.path.join(docs_folder, file_name)
        if file_name.lower().endswith('.pdf'):
            # Extraer texto de PDF usando pypdf
            pdf_text = extract_text_from_pdf(file_path)
            documents.append(Document(text=pdf_text))
        else:
            # Para otros tipos de archivo se utiliza SimpleDirectoryReader
            documents.extend(SimpleDirectoryReader(docs_folder).load_data())

    print(f"Se indexaron {len(documents)} documentos.")

    # Usar DeepInfraEmbeddingModel para el índice
    index = VectorStoreIndex.from_documents(documents, embed_model=embedding_model)
    # Crear un motor de consulta en streaming con un top-k de similitud de 5
    query_engine = index.as_query_engine(streaming=True, similarity_top_k=5)

    # Guardamos el motor de consulta en un archivo con pickle
    # El "wb" es para el modo de escritura binaria
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(query_engine, f)
    
    return query_engine

# Cargar el índice y el modelo de embeddings
query_engine = load_index()

# Configuración de la conversación
#prompt = "Eres un asistente experto en emergencias llamado AsistAI. Responde únicamente preguntas relacionadas con la información en los documentos proporcionados. Si es necesario, complementa con información de internet, pero solo si está estrictamente relacionada con los documentos. Da respuestas claras, precisas y directas, sin explicaciones innecesarias. Intenta dar siempre la solución al problema planteado con una respuesta concisa. Si te piden un número posiblemente sea el de emergencia"

# opcion 2
#promt = "Eres un asistente experto en emergencias. Solo puedes responder preguntas relacionadas con la información contenida en los documentos proporcionados. Si la consulta no está dentro de estos temas, responde educadamente que no puedes ayudar.Si la respuesta requiere información adicional para ser más precisa o actualizada, puedes buscar en internet, pero solo si está estrictamente relacionada con los temas cubiertos en los documentos. No generes respuestas con información no verificada o fuera de contexto.Responde de manera clara, precisa y útil, sin hacer referencia explícita a los documentos en tus respuestas."

prompt = """
Eres un asistente experto en emergencias llamado AsistAI. Responde únicamente preguntas relacionadas con la información contenida en los documentos proporcionados. Si la pregunta no está cubierta por los documentos, indica que no puedes responder. Si es necesario, puedes complementar con información de internet, pero solo si está estrictamente relacionada con los documentos.
Cuando respondas, no hagas referencias al historial de la conversación a menos que sea necesario para la respuesta actual. Concédele mayor importancia a la consulta más reciente. Si te piden un número, es probable que sea un número de emergencia.
Las respuestas deben ser claras, precisas, directas y breves. No incluyas explicaciones innecesarias.
"""

# Saludo
saludo = "👋 ¡Hola! Soy **AsistAI**, tu asistente en situaciones de emergencia. Estoy aquí para ayudarte a resolver cualquier urgencia o aprender qué hacer en momentos críticos. ¿En qué puedo ayudarte?"

# Inicializar historial de conversación en la sesión de Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": prompt},
        {"role": "assistant", "content": saludo}
    ]

# Inicializar el estado de la sesión
if "processing" not in st.session_state:
    st.session_state.processing = False  # Variable de bloqueo para evitar múltiples consultas simultáneas

# Configuración de la aplicación
st.title("🚨 Asistente de Emergencia")
st.write("Pregunta sobre emergencias y obtén respuestas basadas en documentos.")

# Sidebar de la aplicación
with st.sidebar:
    # Botón "Nuevo chat" para reiniciar la conversación
    if st.button("Nuevo chat"):
        st.session_state.messages = [
            {"role": "system", "content": prompt},
            {"role": "assistant", "content": saludo}
        ]
        st.session_state.processing = False
    
    # Información sobre la aplicación
    st.markdown("---")
    st.markdown("### Descripción")
    st.write("Este amable asistente de emergencia proporciona respuestas a preguntas basadas en documentos.")
    st.write("Hecho con ❤️ por los estudiantes de **Accenture**")
    st.markdown("---")

# Mostrar historial de chat
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user", avatar="🧑‍💼").write(f"{msg['content']}")
    elif msg["role"] == "assistant":
        st.chat_message("assistant", avatar="🤖").write(f"{msg['content']}")
    
# Entrada del usuario
if user_query := st.chat_input("Escribe tu pregunta...", disabled=st.session_state.processing):
    # Bloquear entrada de usuario durante la consulta
    st.session_state.processing = True
    
    # Agregar la pregunta al historial de mensajes
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Mostrar la pregunta en el chat
    st.chat_message("user", avatar="🧑‍💼").write(f"{user_query}")
    
    # Recargar la interfaz para reflejar el cambio
    st.rerun()
    
    
# Procesar respuesta en una iteración separada después de que la UI ya se haya actualizado
if st.session_state.processing and len(st.session_state.messages) > 1 and st.session_state.messages[-1]["role"] == "user":
        # Realizar consulta de respuesta en una iteración separada
        with st.chat_message("assistant", avatar="🤖"):
            # Crear un contexto con las preguntas y respuestas previas
            # Limitar historial a los últimos 5 mensajes
            conversation_history = "\n".join(
                [f"(msg['role'].capitalize()): {msg['content']}" for msg in st.session_state.messages]
            )
            
            # Colocamos el spinner en un contenedor vacío para que se pueda manipular
            spinner_placeholder = st.empty()
            # Colocamos el contenedor de respuesta en un contenedor vacío para que se pueda manipular
            response_placeholder = st.empty()
            # Variable para guardar la respuesta
            response_text = ""

            # Mostrar spinner junto al avatar
            with spinner_placeholder:
                with st.spinner("Pensando..."):
                # Generar respuesta en streaming
                    streaming_response = query_engine.query(f"Historial de conversación:\n{conversation_history}\n\nNueva pregunta: {user_query}")
                    
                    # Quitar el spinner después de completar la operación
                    spinner_placeholder.empty()
                    for fragment in streaming_response.response_gen:
                        response_text += fragment  # Acumular texto generado
                        response_placeholder.write(response_text)  # Mostrarlo 

            # Asignar el texto completo como respuesta final
            response = response_text

        # Agregar la respuesta al historial de mensajes
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Desbloquear entrada de usuario después de obtener la consulta
        st.session_state.processing = False
        # Recargar la interfaz para reflejar el cambio
        st.rerun()
