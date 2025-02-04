# AsistencIA

* data (modelo de peleas y accidentes de coche): https://drive.google.com/drive/folders/17B-VwveDxqi3yTAORRlwEvTp96-YSO4s?usp=sharing
* modelos: https://drive.google.com/drive/folders/1gOx2LQRSvPosBatWlzqP7ncQUejXaGAI?usp=sharing

_AsistencIA_ es un proyecto de Inteligencia Artificial y Big Data orientado a la detección temprana de tres tipos de emergencias mediante análisis de video: accidentes de coche, incendios y peleas. La idea principal es utilizar cámaras de la vía pública para alertar de forma inmediata a servicios de emergencia (bomberos, ambulancias y policía) y, a la vez, ofrecer herramientas de capacitación ciudadana y asistencia en tiempo real.

<img src="img/logo.png" alt="AsistencIA" width="400"/>

## Justificación y descripción del proyecto

El proyecto _AsistencIA_ tiene como objetivo desarrollar un sistema integral que detecte, mediante análisis de video, situaciones críticas en tiempo real. Las principales emergencias a detectar son:

- **Accidentes de coche**
- **Incendios**
- **Peleas**

Además, se incorporan funcionalidades adicionales para mejorar la respuesta y formación de los ciudadanos ante situaciones de emergencia:

- **Módulo de Capacitación:** Permite subir enlaces a videos (por ejemplo, de YouTube) relacionados con primeros auxilios, para los cuales se genera un resumen y un cuestionario interactivo, facilitando el aprendizaje y la capacitación.
- **Chatbot Asistencial:** Un asistente conversacional que responde preguntas sobre cómo actuar en situaciones de emergencia, ofreciendo instrucciones claras y, de ser necesario, generando imágenes ilustrativas para reforzar la explicación.

## Funcionalidades

1. **Detección en tiempo real:**
   - *Webcam:* Permite probar el sistema en vivo a través de la cámara web.
   - *Carga de Video:* Permite subir archivos de video y obtener una predicción en tiempo real de la probabilidad de que se esté produciendo alguno de los eventos críticos.

2. **Capacitación ciudadana:**
   - *Resumen y Cuestionario:* Al subir un enlace a un video de YouTube o un video local relacionado con primeros auxilios, el sistema genera un resumen del contenido y un cuestionario interactivo para reforzar el aprendizaje.

3. **Asistencia mediante chatbot:**
   - El chatbot responde a consultas relacionadas con protocolos de actuación en caso de emergencias (accidentes, incendios, peleas), ofreciendo directrices claras y generando imágenes de ejemplo para ilustrar las instrucciones.


## Aplicación web

La aplicación web se desarrollará utilizando **Streamlit** y contará con tres módulos principales:

1. **Detección en Tiempo Real:**  
   - Interfaz para uso de la webcam y subida de videos para análisis en tiempo real.
2. **Capacitación Ciudadana:**  
   - Subida de enlaces o videos para generar resúmenes y cuestionarios.
3. **Chatbot Asistencial:**  
   - Asistente conversacional que ofrece recomendaciones y guía en situaciones de emergencia.


## Ejemplo de uso con Agentes (con n8n):
### Detección y gestión de un accidente de coche:

1. **Captura del evento en tiempo real:**
   - Una cámara de la vía pública graba un accidente de coche.
   - Un agente en n8n recibe el video mediante un webhook y lo envía a un servicio de análisis que evalúa la probabilidad de emergencia.

2. **Evaluación y confirmación:**
   - Si el análisis detecta una probabilidad alta (por ejemplo, >70%), se considera que se ha producido una emergencia.
   - Se activa una alerta automática que notifica a servicios de emergencia (ambulancias, policía, etc.) mediante correo, SMS o sistemas centralizados.

3. **Registro y seguimiento:**
   - El evento se almacena en una base de datos para un control histórico y análisis posterior.
   - La aplicación web (Streamlit) recibe la notificación y muestra la información en tiempo real a los encargados y a la comunidad, incluyendo la ubicación y detalles del accidente.

