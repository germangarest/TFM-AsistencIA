# 🏷️ Índice
1. [🔎 Justificación y descripción del proyecto](#1-justificación-y-descripción-del-proyecto)  
2. [🗂️ Obtención de datos](#2-obtención-de-datos)  
3. [📊 Descripción de los datos](#3-descripción-de-los-datos)  
4. [📈 Exploración y visualización de los datos](#4-exploración-y-visualización-de-los-datos)  
5. [🔧 Preparación de los datos para los algoritmos de Machine Learning](#5-preparación-de-los-datos-para-los-algoritmos-de-machine-learning)  
6. [🏋️ Entrenamiento del modelo y comprobación del rendimiento](#6-entrenamiento-del-modelo-y-comprobación-del-rendimiento)  
7. [🗣️ Se tiene que incluir alguna de las técnicas estudiadas en el tema de Procesamiento de Lenguaje Natural](#7-se-tiene-que-incluir-alguna-de-las-técnicas-estudiadas-en-el-tema-de-procesamiento-de-lenguaje-natural)  
8. [🌐 Aplicación web](#8-aplicación-web)  
9. [💡 Conclusiones](#9-conclusiones)

---

## 1. Justificación y descripción del proyecto
_AsistencIA_ es un proyecto de Inteligencia Artificial y Big Data orientado a la detección temprana de tres tipos de emergencias mediante análisis de video: accidentes de coche, incendios y peleas. La idea principal es utilizar cámaras de la vía pública para alertar de forma inmediata a servicios de emergencia (bomberos, ambulancias y policía) y, a la vez, ofrecer herramientas de capacitación ciudadana y asistencia en tiempo real.

<img src="img/logo.png" alt="AsistencIA" width="400"/>

El proyecto _AsistencIA_ tiene como objetivo desarrollar un sistema integral que detecte, mediante análisis de video, situaciones críticas en tiempo real. Las principales emergencias a detectar son:

- **Accidentes de coche**  
- **Incendios**  
- **Peleas**

Además, se incorporan funcionalidades adicionales para mejorar la respuesta y formación de los ciudadanos ante situaciones de emergencia:

- **Módulo de Capacitación:** Permite subir enlaces a videos (por ejemplo, de YouTube) relacionados con primeros auxilios, para los cuales se genera un resumen y un cuestionario interactivo, facilitando el aprendizaje y la capacitación.
- **Chatbot Asistencial:** Un asistente conversacional que responde preguntas sobre cómo actuar en situaciones de emergencia, ofreciendo instrucciones claras y, de ser necesario, generando imágenes ilustrativas para reforzar la explicación.

---

## 2. Obtención de datos
### ACCIDENTES DE COCHE Y PELEAS:
En cuanto al tema de los datos, hubo bastantes problemas con los modelos de accidentes de coche y peleas. Primero, comenzamos probando un dataset de imágenes, las cuales habían sido extraídas de los videos respectivos, el cual era de [Kaggle](https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset). Entrenamos el modelo y daba una alta precisión, pero al probarlo en la aplicación web de Streamlit con otros videos, daba precisiones del 100% en cualquier parte del video, incluso si no se veía un accidente ni una pelea, por lo que después de muchos intentos de optimizaciones, pasamos a entrenar el modelo con el mismo dataset pero directamente con los videos, el cual estaba en la página oficial [University of Central Florida](https://www.crcv.ucf.edu/projects/real-world/).

Una vez más, el modelo no respondía bien, aunque esta vez directamente daba una precisión bajísima, por lo que el dataset era inútil. Finalmente, nos dimos cuenta de que el problema era que los videos de peleas (o accidentes) eran de varios minutos en los que no solo se mostraba el incidente, sino que el incidente duraba 5 o 10 segundos y el resto era una situación normal, por lo que el modelo no sabía diferenciar qué era una situación anómala.

Después de todos estos problemas, finalmente, nos dimos cuenta de que la mejor manera para entrenar un modelo de estas características era coger clips cortos donde solo se muestre el incidente, y que el modelo aprenda las características y parámetros para la predicción.

#### DATASET USADO ACCIDENTES DE COCHE:
Finalmente, para el modelo de accidentes de coche usamos el [CarCrashDataset en Github](https://github.com/Cogito2012/CarCrashDataset). Estos clips han sido obtenidos de grabaciones de cámaras en el salpicadero de coches (dashcam). El dataset incluye diferentes situaciones medioambientales para una mejor variabilidad.

#### DATASET USADO PELEAS:
Para el modelo de peleas, usamos el [RWF-2000 de Hugging Face](https://huggingface.co/datasets/DanJoshua/RWF-2000). Estos videos han sido sacados de cámaras de vigilancia.

### INCENDIOS:

---

## 3. Descripción de los datos
Se debe dar una descripción completa de los datos indicando qué significa cada uno de los atributos.

### ACCIDENTES DE COCHE:
El dataset se divide en dos carpetas:
* **Normal_Videos_for_Event_Recognition**: videos sin accidentes de coche.
* **CrashAccidents**: videos de accidentes de coche.

Atributos de los videos (metadatos):
* **Resolución (ancho y alto)**: 1280×720 px  
* **Frames por segundo (fps)**: 10 fps  
* **Número de frames**: 50 frames  
* **Duración**: 5 segundos  
* **Tamaño del archivo**: varía de 0.5 MB a 8 MB  
* **Formato de video**: .mp4  

### PELEAS:
El dataset se divide en dos carpetas:
* **NonFight**: videos sin peleas.
* **Fight**: videos de peleas.

Atributos de los videos (metadatos):
* **Resolución (ancho y alto)**: 640×360 px  
* **Frames por segundo (fps)**:  
* **Número de frames**:  
* **Duración**: de 5 a 10 segundos  
* **Tamaño del archivo**:  
* **Formato de video**: .avi  

### INCENDIOS:

---

## 4. Exploración y visualización de los datos
### ACCIDENTES DE COCHE:
Algunos ejemplos de videos de accidentes:

Algunos ejemplos de situaciones normales:

### PELEAS:
Algunos ejemplos de videos de peleas:

Algunos ejemplos de situaciones normales:

### INCENDIOS:

---

## 5. Preparación de los datos para los algoritmos de Machine Learning
### ACCIDENTES DE COCHE Y PELEAS:

### INCENDIOS:

---

## 6. Entrenamiento del modelo y comprobación del rendimiento
### ACCIDENTES DE COCHE Y PELEAS:

### INCENDIOS:

---

## 7. Se tiene que incluir alguna de las técnicas estudiadas en el tema de Procesamiento de Lenguaje Natural

---

## 8. Aplicación web
La aplicación web se desarrollará utilizando **Streamlit** y contará con tres módulos principales:

1. **Detección en Tiempo Real:**  
   - Interfaz para uso de la webcam y subida de videos para análisis en tiempo real.
2. **Capacitación Ciudadana:**  
   - Subida de enlaces o videos para generar resúmenes y cuestionarios.
3. **Chatbot Asistencial:**  
   - Asistente conversacional que ofrece recomendaciones y guía en situaciones de emergencia.
---

## 9. Conclusiones
