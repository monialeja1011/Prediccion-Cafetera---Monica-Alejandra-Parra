# Predicción de la Producción de Café en Cartago, Valle del Cauca

## 1. Nombre del proyecto

**Predicción de la Producción de Café en Cartago, Valle del Cauca**


## 2. Descripción del proyecto

Este proyecto de Inteligencia Artificial tiene como propósito analizar datos agrícolas del departamento del Valle del Cauca y desarrollar un modelo de Machine Learning capaz de estimar la producción de café.

El proyecto se enfoca especialmente en el municipio de Cartago, Valle del Cauca.

Para desarrollar el sistema se utiliza Python junto con herramientas de análisis y aprendizaje automático como Pandas, NumPy, Matplotlib y Scikit-learn.

El proceso incluye la carga y limpieza de los datos, el filtrado de los registros correspondientes al cultivo de café, el análisis exploratorio, el entrenamiento de un modelo de regresión y finalmente la generación de estimaciones de producción.


## 3. Problemática

La producción agrícola puede variar entre diferentes periodos debido a factores relacionados con el área sembrada, el área cosechada y las características de cada municipio.

Analizar los datos históricos permite identificar patrones y variaciones en la producción.

En este proyecto se busca utilizar información agrícola disponible para construir un modelo de Machine Learning que permita realizar estimaciones de producción de café para Cartago.

La estimación generada por el modelo debe entenderse como un resultado calculado a partir de los datos suministrados y no como una medición real de producción futura.



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



## 9. Análisis de los datos

El análisis se realiza mediante:

```text
src/analizar_datos.py
```

Este script permite revisar información relacionada con los datos procesados y obtener información útil para comprender el conjunto de datos antes de entrenar el modelo.



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



## 11. Modelo de Machine Learning

Para realizar la estimación se utiliza el algoritmo:

**Random Forest Regressor**

Random Forest es un algoritmo de Machine Learning basado en múltiples árboles de decisión.

En este proyecto se utiliza para realizar una predicción de tipo regresión, donde la variable que se busca estimar corresponde a la producción de café.



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


## 15. Gráficos generados

El proyecto genera diferentes archivos gráficos:

* **Producción histórica:** `data/produccion_cartago.png`. Permite visualizar información relacionada con la producción de café de Cartago.
* **Predicción:** `data/prediccion_cartago.png`. Representa gráficamente el resultado de la estimación realizada por el modelo.


## 16. Archivos principales

* `data/eva_base_agricola.csv`: contiene los datos agrícolas utilizados como fuente inicial para el proyecto.
* `data/cafe_valle_limpio.csv`: contiene los registros procesados correspondientes al cultivo de café.
* `data/modelo_cafetero.pkl`: contiene el modelo Random Forest entrenado.
* `src/limpiar_datos.py`: realiza el procesamiento inicial de los datos.
* `src/analizar_datos.py`: permite analizar los datos procesados.
* `src/graficar_produccion.py`: genera gráficos relacionados con la producción.
* `src/entrenar_modelo.py`: entrena y evalúa el modelo Random Forest.
* `src/predecir.py`: carga el modelo entrenado y permite realizar estimaciones.
* `src/graficar_prediccion.py`: genera la visualización de la predicción.



## 17. Instalación

Clonar o descargar el proyecto y acceder a la carpeta:

```bash
cd prediccion-cafetera
```

Crear el entorno virtual:

```bash
python3 -m venv .venv
```

Activar el entorno virtual.

### En Linux/macOS

```bash
source .venv/bin/activate
```

### En Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

Instalar las dependencias:

```bash
pip install -r requirements.txt
```


## 18. Ejecución

Ejecutar los scripts en el siguiente orden secuencial para procesar los datos, entrenar el modelo y visualizar los resultados.

### 1. Limpiar los datos

```bash
python src/limpiar_datos.py
```

### 2. Analizar los datos

```bash
python src/analizar_datos.py
```

### 3. Generar gráfico de producción

```bash
python src/graficar_produccion.py
```

### 4. Entrenar el modelo

```bash
python src/entrenar_modelo.py
```

### 5. Realizar una predicción

```bash
python src/predecir.py
```

### 6. Generar gráfico de predicción

```bash
python src/graficar_prediccion.py
```



## 19. Resultados principales

A continuación se presenta un resumen con las métricas y datos clave obtenidos durante el desarrollo del proyecto:

| Resultado                  |            Valor |
| -------------------------- | ---------------: |
| Registros de café          |              273 |
| Municipios                 |               39 |
| Periodo                    |        2019–2025 |
| Registros de entrenamiento |              218 |
| Registros de prueba        |               55 |
| MAE                        | 305.08 toneladas |
| RMSE                       | 528.10 toneladas |
| R²                         |           0.8650 |
| Escenario de predicción    |    Cartago, 2026 |
| Producción estimada        | 327.36 toneladas |



## 20. Control de versiones

El proyecto utiliza Git para registrar los cambios realizados durante el desarrollo.

Git permite mantener un historial de las modificaciones realizadas en los archivos y facilita la recuperación de versiones anteriores.

El proyecto es almacenado y compartido de forma pública mediante GitHub.



## 21. Consideraciones

La predicción generada por el modelo depende estrictamente de los datos utilizados durante el entrenamiento y de las variables ingresadas al momento de realizar la estimación.

Por esta razón, los resultados deben interpretarse únicamente como estimaciones realizadas mediante Machine Learning y no como valores oficiales o definitivos de producción futura.

La incorporación de nuevos datos y variables agrícolas podría permitir mejorar posteriormente el modelo.


## 22. Próximos pasos

Como trabajo futuro para optimizar el sistema se plantea:

* Incorporar nuevas variables agrícolas, como clima, tipo de suelo y fertilizantes.
* Evaluar otros algoritmos de Machine Learning, como XGBoost y Support Vector Regression.
* Comparar de forma directa diferentes modelos de regresión.
* Realizar el ajuste fino de hiperparámetros.
* Analizar a fondo los errores de las predicciones.
* Incorporar nuevos datos históricos a medida que estén disponibles.
* Mejorar la interfaz de interacción con el modelo, por ejemplo mediante una interfaz web con Streamlit.
* Evaluar periódicamente el desempeño del modelo.


## 23. Conclusión

Este proyecto demuestra la aplicación práctica de técnicas de Inteligencia Artificial y Machine Learning al análisis de datos agrícolas.

A partir de datos del Valle del Cauca se realizó un proceso completo de limpieza, análisis y preparación de información relacionada con el cultivo de café.

Posteriormente se entrenó un modelo **Random Forest Regressor**, obteniendo un MAE de 305.08 toneladas, un RMSE de 528.10 toneladas y un R² de 0.8650 sobre el conjunto de prueba utilizado.

Finalmente, se implementó una etapa de predicción para el municipio de Cartago. Utilizando un escenario simulado para 2026 con 280 hectáreas sembradas y 240 hectáreas cosechadas, el modelo estimó una producción de 327.36 toneladas.

El proyecto constituye una primera versión funcional de un sistema de estimación de producción agrícola y queda abierto a futuras mejoras técnicas.
