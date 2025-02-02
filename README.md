# AsistencIA

_AsistencIA_ es un proyecto de Inteligencia Artificial y Big Data orientado a la detección temprana de tres tipos de emergencias mediante análisis de video: accidentes de coche, incendios y peleas. La idea principal es utilizar cámaras de la vía pública para alertar de forma inmediata a servicios de emergencia (bomberos, ambulancias y policía) y, a la vez, ofrecer herramientas de capacitación ciudadana y asistencia en tiempo real.

## Índice

- [Justificación y descripción del Proyecto](#descripción-del-proyecto)
- [Funcionalidades](#funcionalidades)
- [Obtención y Preparación de Datos](#obtención-y-preparación-de-datos)
- [Entrenamiento del Modelo y Evaluación del Rendimiento](#entrenamiento-del-modelo-y-evaluación-del-rendimiento)
- [Aplicación Web](#aplicación-web)
- [Aplicaciones de Procesamiento de Lenguaje Natural](#aplicaciones-de-procesamiento-de-lenguaje-natural)
- [Estructura del Repositorio](#estructura-del-repositorio)
- [Conclusiones](#conclusiones)
- [Instalación y Ejecución](#instalación-y-ejecución)
- [Créditos y Licencia](#créditos-y-licencia)

## Justificación y descripción del Proyecto

El proyecto _AsistencIA_ tiene como objetivo desarrollar un sistema integral que detecte, mediante análisis de video, situaciones críticas en tiempo real. Las principales emergencias a detectar son:

- **Accidentes de coche**
- **Incendios**
- **Peleas**

Además, se incorporan funcionalidades adicionales para mejorar la respuesta y formación de los ciudadanos ante situaciones de emergencia:

- **Módulo de Capacitación:** Permite subir enlaces a videos (por ejemplo, de YouTube) relacionados con primeros auxilios, para los cuales se genera un resumen y un cuestionario interactivo, facilitando el aprendizaje y la capacitación.
- **Chatbot Asistencial:** Un asistente conversacional que responde preguntas sobre cómo actuar en situaciones de emergencia, ofreciendo instrucciones claras y, de ser necesario, generando imágenes ilustrativas para reforzar la explicación.

## Funcionalidades

1. **Detección en Tiempo Real:**
   - *Webcam:* Permite probar el sistema en vivo a través de la cámara web.
   - *Carga de Video:* Permite subir archivos de video y obtener una predicción en tiempo real de la probabilidad de que se esté produciendo alguno de los eventos críticos.

2. **Capacitación Ciudadana:**
   - *Resumen y Cuestionario:* Al subir un enlace a un video de YouTube o un video local relacionado con primeros auxilios, el sistema genera un resumen del contenido y un cuestionario interactivo para reforzar el aprendizaje.

3. **Asistencia mediante Chatbot:**
   - El chatbot responde a consultas relacionadas con protocolos de actuación en caso de emergencias (accidentes, incendios, peleas), ofreciendo directrices claras y generando imágenes de ejemplo para ilustrar las instrucciones.


## Obtención y Preparación de Datos

- **Fuentes de Datos:**  
  - Cámaras de videovigilancia instaladas en la vía pública.
  - Bases de datos públicas y privadas relacionadas con incidentes viales, incendios y altercados.
  - Datos adicionales de videos educativos y de emergencias.
- **Métodos de Obtención:**  
  - Captura en tiempo real mediante APIs de cámaras.
  - Técnicas de scrapping para la recolección de datos de plataformas públicas.
  - Encuestas y colaboraciones con organismos oficiales.
- **Proceso de Limpieza:**  
  - Eliminación de registros nulos y erróneos.
  - Normalización y etiquetado de los datos para facilitar su posterior análisis.

## Entrenamiento del Modelo y Evaluación del Rendimiento


## Aplicación Web

La aplicación web se desarrollará utilizando **Streamlit** y contará con tres módulos principales:

1. **Detección en Tiempo Real:**  
   - Interfaz para uso de la webcam y subida de videos para análisis en tiempo real.
2. **Capacitación Ciudadana:**  
   - Subida de enlaces o videos para generar resúmenes y cuestionarios.
3. **Chatbot Asistencial:**  
   - Asistente conversacional que ofrece recomendaciones y guía en situaciones de emergencia.

## Aplicaciones de Procesamiento de Lenguaje Natural

El proyecto integra técnicas avanzadas de NLP para:

- **Reconocimiento de Voz y Síntesis de Texto a Voz:**  
  - Facilitar la interacción del usuario con el sistema mediante comandos de voz y respuestas audibles.
- **Generación de Resúmenes y Cuestionarios:**  
  - Procesar videos educativos para extraer los puntos clave y generar preguntas que refuercen el aprendizaje.
- **Chatbot Interactivo:**  
  - Responder de manera precisa a consultas sobre protocolos de actuación en emergencias, utilizando modelos de lenguaje entrenados en datos específicos del dominio.
