# AIVA_2026: RBC Counter

## Descripción del proyecto

Este repositorio contiene el desarrollo del sistema **RBC_Counter**, una aplicación de visión artificial diseñada para la detección, segmentación y conteo automático de glóbulos rojos en imágenes de plasma sanguíneo obtenidas mediante microscopio.

El objetivo del sistema es automatizar el proceso de conteo manual realizado habitualmente por técnicos de laboratorio, reduciendo el tiempo de análisis y manteniendo un nivel de precisión comparable al conteo humano.

El sistema analizará imágenes capturadas en condiciones controladas, detectará los glóbulos rojos presentes y generará un resultado estructurado que incluye el número total de células detectadas y su localización mediante *bounding boxes*.

---

## Objetivos del sistema

Los objetivos principales del proyecto son:

* Automatizar el conteo de glóbulos rojos en imágenes microscópicas.
* Detectar y segmentar cada célula mediante *bounding boxes*.
* Filtrar elementos no relevantes para el análisis (por ejemplo células azules).
* Generar un archivo estructurado con los resultados del procesamiento.
* Permitir la supervisión posterior mediante imágenes con segmentación visual.

---

## Funcionamiento general

El sistema seguirá el siguiente flujo de procesamiento:

1. Recepción de una imagen procedente del microscopio.
2. Procesamiento de la imagen mediante técnicas de visión artificial.
3. Detección de glóbulos rojos presentes en la muestra.
4. Generación de *bounding boxes* para cada célula detectada.
5. Conteo automático del número total de glóbulos rojos.
6. Generación de un archivo XML con los resultados.
7. Almacenamiento de:

   * imagen original
   * archivo XML generado
   * imagen con las *bounding boxes* superpuestas.

---

## Plataforma de ejecución

El sistema está diseñado para ejecutarse en el siguiente entorno:

* **Hardware:** Raspberry Pi 5
* **Sistema operativo:** Raspberry Pi OS (basado en Debian)
* **Lenguaje principal:** Python

---

## Tecnologías utilizadas

El desarrollo del proyecto utilizará principalmente las siguientes herramientas:

* **Python**
* **OpenCV** para procesamiento de imágenes
* **NumPy** para operaciones numéricas
* **Matplotlib** para visualización de resultados
* **PyTest / unittest** para pruebas automáticas

---

## Estructura del repositorio

El repositorio está organizado de la siguiente forma:

```
AIVA_2026:RBC_Counter

src/
    Implementación del sistema de detección y conteo

tests/
    Tests unitarios del sistema

mockups/
    Mockups del sistema principal

docs/
    Documentación del proyecto
```

---

## Estado del proyecto

Actualmente el proyecto se encuentra en fase de **definición de requisitos y planificación**, siguiendo el estándar **IEEE/ANSI 29148** para la especificación de requisitos del sistema.

Las tareas de desarrollo están organizadas mediante **issues de GitHub**, donde se definen las funcionalidades a implementar y el tiempo estimado para cada una.

---

## Autores

Daniel Cámara Núñez - Carla Barbero Martín

Máster en Inteligencia Artificial

Asignatura: Aplicaciones Industriales de Visión Artificial

Universidad Rey Juan Carlos

