# Predicción de la Producción de Café en Cartago, Valle del Cauca

## 1. Nombre del proyecto

**Predicción de la Producción de Café en Cartago, Valle del Cauca**

---

## 2. Descripción del proyecto

Este proyecto de Inteligencia Artificial tiene como propósito analizar datos agrícolas del departamento del Valle del Cauca y desarrollar un modelo de Machine Learning capaz de estimar la producción de café.

El proyecto se enfoca especialmente en el municipio de Cartago, Valle del Cauca.

Para desarrollar el sistema se utiliza Python junto con herramientas de análisis y aprendizaje automático como Pandas, NumPy, Matplotlib y Scikit-learn.

El proceso incluye la carga y limpieza de los datos, el filtrado de los registros correspondientes al cultivo de café, el análisis exploratorio, el entrenamiento de un modelo de regresión y finalmente la generación de estimaciones de producción.

---

## 3. Problemática

La producción agrícola puede variar entre diferentes periodos debido a factores relacionados con el área sembrada, el área cosechada y las características de cada municipio.

Analizar los datos históricos permite identificar patrones y variaciones en la producción.

En este proyecto se busca utilizar información agrícola disponible para construir un modelo de Machine Learning que permita realizar estimaciones de producción de café para Cartago.

La estimación generada por el modelo debe entenderse como un resultado calculado a partir de los datos suministrados y no como una medición real de producción futura.

---

## 4. Objetivo general

Desarrollar un modelo de Inteligencia Artificial que permita estimar la producción de café en Cartago, Valle del Cauca, utilizando información histórica agrícola.

### Objetivos específicos

* Obtener y organizar datos agrícolas.
* Identificar los registros correspondientes al cultivo de café.
* Limpiar y preparar los datos.
* Analizar el comportamiento histórico de la producción.
* Generar visualizaciones de los datos.
* Preparar las variables utilizadas por el modelo.
* Entrenar un modelo de Machine Learning.
* Evaluar el desempeño del modelo.
* Realizar una estimación de producción para Cartago.

---

## 5. Datos utilizados

El proyecto utiliza información agrícola contenida en el archivo:

```text
data/eva_base_agricola.csv
```

Los datos fueron procesados para identificar los registros relacionados con el cultivo de café.

Después del proceso de limpieza se obtuvo el archivo:

```text
data/cafe_valle_limpio.csv
```

### Resultados del procesamiento

* 273 registros correspondientes al cultivo de café.
* 39 municipios representados.
* Periodo disponible: 2019–2025.

Los datos utilizados corresponden a información agrícola y son empleados para realizar análisis y entrenar el modelo.

---

## 6. Tecnologías utilizadas

El proyecto utiliza:

* **Python:** lenguaje principal.
* **Pandas:** carga y manipulación de datos.
* **NumPy:** operaciones numéricas.
* **Matplotlib:** generación de gráficos.
* **Scikit-learn:** desarrollo del modelo de Machine Learning.
* **Git:** control de versiones.
* **GitHub:** almacenamiento del proyecto.

Las dependencias utilizadas se encuentran en:

```text
requirements.txt
```

---

## 7. Estructura del proyecto

```text
prediccion-cafetera/
│
├── data/
│   ├── cafe_valle_limpio.csv
│   ├── eva_base_agricola.csv
│   ├── modelo_cafetero.pkl
│   ├── prediccion_cartago.png
│   └── produccion_cartago.png
│
├── src/
│   ├── analizar_datos.py
│   ├── entrenar_modelo.py
│   ├── graficar_prediccion.py
│   ├── graficar_produccion.py
│   ├── limpiar_datos.py
│   └── predecir.py
│
├── notebooks/
├── tests/
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 8. Procesamiento de los datos

El procesamiento se realiza mediante el script:

```text
src/limpiar_datos.py
```

Este proceso permite cargar la información agrícola y seleccionar los registros correspondientes al cultivo de café.

El resultado del procesamiento se almacena en:

```text
data/cafe_valle_limpio.csv
```

---

## 9. Análisis de los datos

El análisis se realiza mediante:

```text
src/analizar_datos.py
```

Este script permite revisar información relacionada con los datos procesados y obtener información útil para comprender el conjunto de datos antes de entrenar el modelo.

---

## 10. Visualización

El proyecto genera gráficos utilizando Matplotlib.

El script:

```text
src/graficar_produccion.py
```

permite generar una visualización relacionada con la producción histórica de Cartago.

El resultado se almacena en:

```text
data/produccion_cartago.png
```

También se genera una gráfica relacionada con la estimación:

```text
data/prediccion_cartago.png
```

utilizando:

```text
src/graficar_prediccion.py
```

---

## 11. Modelo de Machine Learning

Para realizar la estimación se utiliza el algoritmo:

**Random Forest Regressor**

Random Forest es un algoritmo de Machine Learning basado en múltiples árboles de decisión.

En este proyecto se utiliza para realizar una predicción de tipo regresión, donde la variable que se busca estimar corresponde a la producción de café.

---

## 12. Entrenamiento del modelo

El entrenamiento se realiza mediante:

```text
src/entrenar_modelo.py
```

Los datos disponibles fueron divididos en dos grupos:

* **Datos de entrenamiento:** 218 registros.
* **Datos de prueba:** 55 registros.

Los datos de entrenamiento son utilizados para que el modelo aprenda las relaciones existentes entre las variables.

Los datos de prueba permiten evaluar el comportamiento del modelo utilizando registros que no fueron utilizados directamente durante el entrenamiento.

El modelo entrenado se almacena en:

```text
data/modelo_cafetero.pkl
```

---

## 13. Evaluación del modelo

El modelo Random Forest obtuvo los siguientes resultados sobre el conjunto de prueba:

| Métrica |        Resultado |
| ------- | ---------------: |
| MAE     | 305.08 toneladas |
| RMSE    | 528.10 toneladas |
| R²      |           0.8650 |

### Glosario de métricas

* **MAE (Error Absoluto Medio):** representa el promedio de la diferencia absoluta entre las predicciones y los valores reales. En este caso es de 305.08 toneladas.
* **RMSE (Raíz del Error Cuadrático Medio):** mide el error de las predicciones dando mayor peso a los errores grandes. En este caso es de 528.10 toneladas.
* **R² (Coeficiente de determinación):** permite medir qué proporción de la variabilidad de la variable objetivo explica el modelo sobre el conjunto de prueba utilizado. El resultado obtenido fue de 0.8650.

Estos resultados corresponden al conjunto de prueba utilizado durante el entrenamiento y evaluación del modelo.

---

## 14. Predicción para Cartago

La generación de predicciones se realiza mediante:

```text
src/predecir.py
```

El programa permite ingresar información como:

* Municipio.
* Año.
* Hectáreas sembradas.
* Hectáreas cosechadas.

Para realizar una demostración se utilizó el siguiente escenario de prueba:

```text
Municipio: Cartago
Año: 2026
Hectáreas sembradas: 280
Hectáreas cosechadas: 240
```

Con estas condiciones, el modelo obtuvo:

**Producción estimada: 327.36 toneladas**

**Nota:** Este resultado corresponde a una estimación generada por el modelo bajo las condiciones de entrada indicadas. No representa una medición real de producción de Cartago para 2026.

---

## 15. Gráficos generados

El proyecto genera diferentes archivos gráficos:

* **Producción histórica:** `data/produccion_cartago.png`. Permite visualizar información relacionada con la producción de café de Cartago.
* **Predicción:** `data/prediccion_cartago.png`. Representa gráficamente el resultado de la estimación realizada p_
