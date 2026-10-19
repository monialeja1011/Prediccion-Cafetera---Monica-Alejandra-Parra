# Predicción de la producción de café en Cartago, Valle del Cauca

Proyecto de Inteligencia Artificial para analizar registros agrícolas del Valle del Cauca y estimar la producción de café. El municipio de interés es Cartago. El modelo principal es de regresión; para el segundo corte se añadió además un perceptrón binario implementado desde cero con NumPy, como ejercicio académico.

## Objetivo y problemática

La producción agrícola varía entre municipios y periodos. Este proyecto organiza datos históricos de café, explora la relación entre área, año y producción, y entrena un modelo que estima la producción a partir de variables agrícolas disponibles.

La estimación es una salida del modelo basada en los datos y las entradas suministradas; no constituye una medición oficial ni garantiza la producción futura.

## Datos

La fuente incluida en el proyecto es `data/eva_base_agricola.csv`. El script `src/limpiar_datos.py` filtra los registros de café del Valle del Cauca y exporta `data/cafe_valle_limpio.csv`.

El conjunto limpio contiene **273 registros**, **39 municipios** y el periodo **2019–2025**. Conserva estas columnas: `codigo_municipio`, `municipio`, `año`, `hectareas_sembradas`, `hectareas_cosechadas`, `produccion` y `rendimiento`.

### Inspección y limpieza

La base original tiene 166.732 filas y 18 columnas. La inspección con Pandas no encontró nulos ni filas duplicadas. En los registros de café del Valle tampoco se encontraron nulos, duplicados, valores negativos en las medidas ni casos donde el área cosechada superara la sembrada. Los tipos de las variables cuantitativas fueron convertidos explícitamente a numéricos antes de exportar.

El criterio IQR señaló valores estadísticamente extremos en las cuatro medidas revisadas: área sembrada (47), área cosechada (46), producción (40) y rendimiento (27). Se conservaron porque ser atípicos estadísticamente no prueba que sean errores, y las comprobaciones de consistencia no identificaron valores imposibles.

| Estado | Filas | Columnas | Nulos |
|---|---:|---:|---:|
| Base EVA original | 166.732 | 18 | 0 |
| Café del Valle limpio | 273 | 7 | 0 |

La reducción de filas y columnas corresponde al filtro de alcance —café del Valle y variables seleccionadas—. El tratamiento posterior no eliminó ninguno de los 273 registros. El resumen se guarda en `data/resumen_antes_despues.csv`.

## Visualizaciones con Seaborn

El script `src/graficos_seaborn.py` crea las siguientes imágenes en `data/`:

1. **Distribución de la producción** — `grafico_1_distribucion_produccion.png`. La mayor parte de los registros se concentra en producciones menores y la distribución tiene una cola hacia valores altos. Los valores extremos se conservaron porque no se identificaron como errores.
2. **Área cosechada y producción** — `grafico_2_area_produccion.png`. Se observa una relación positiva general: al aumentar el área cosechada suele aumentar la producción. La dispersión entre áreas similares muestra que el área no explica por sí sola la variación.
3. **Producción por año** — `grafico_3_produccion_por_anio.png`. Las medianas fluctúan y hay observaciones altas en todos los años, sin una tendencia anual sostenida. El año aporta información, pero debe considerarse junto con otras variables.

Ejecutar las gráficas con:

```bash
python src/graficos_seaborn.py
```

## Modelos

### Modelo principal: regresión

El proyecto principal estima una variable continua, la producción. `src/entrenar_modelo.py` entrena un `RandomForestRegressor` de Scikit-Learn con año, hectáreas sembradas y hectáreas cosechadas. La división reportada es de 218 registros para entrenamiento y 55 para prueba.

Resultados reportados sobre ese conjunto de prueba:

| Métrica | Resultado |
|---|---:|
| MAE | 305,08 toneladas |
| RMSE | 528,10 toneladas |
| R² | 0,8650 |

El modelo se guarda en `data/modelo_cafetero.pkl`. `src/predecir.py` demuestra una estimación para Cartago en 2026 con 280 hectáreas sembradas y 240 cosechadas: **327,36 toneladas**, según el modelo. Es un escenario de demostración y no un dato observado.

### Ejercicio del segundo corte: perceptrón binario desde cero

`src/perceptron_proyecto.py` implementa con NumPy una neurona con sigmoide, pérdida logarítmica y descenso del gradiente. Para este ejercicio, el objetivo binario se define así:

- `1` si `produccion` es mayor que la mediana (**688,67**).
- `0` si `produccion` es menor o igual a la mediana.

Las clases contienen 136 y 137 registros, respectivamente. Las dos entradas numéricas son `año` y `hectareas_sembradas`, estandarizadas con media y desviación estándar. `produccion` no se usa como entrada porque define la etiqueta; `rendimiento` tampoco se incluye porque está directamente relacionado con la producción. Este ejercicio de clasificación no reemplaza la tarea principal del proyecto, que sigue siendo regresión.

Resultados ejecutados:

| Resultado | Valor |
|---|---:|
| w1 (año estandarizado) | 0,197108 |
| w2 (hectáreas sembradas estandarizadas) | 9,020746 |
| Sesgo (b) | 3,342654 |
| Pérdida inicial | 0,693147 |
| Pérdida final | 0,204417 |
| Exactitud de entrenamiento | 92,67 % |

Los dos pesos positivos indican que, manteniendo las demás condiciones del modelo, aumentar cualquiera de las entradas estandarizadas eleva el logit estimado para la clase 1. La frontera separa gran parte de los registros, pero hay puntos de ambas clases cerca de ella; por tanto, los datos no son perfectamente separables con una frontera lineal. La exactitud corresponde a los mismos datos usados para entrenar y no es una medición independiente de generalización.

El script genera `data/perceptron_curva_aprendizaje.png` y `data/perceptron_frontera_decision.png`. La curva confirma que la pérdida disminuye durante el entrenamiento.

## Estructura del proyecto

```text
prediccion-cafetera/
├── data/
│   ├── eva_base_agricola.csv
│   ├── cafe_valle_limpio.csv
│   ├── resumen_antes_despues.csv
│   ├── grafico_1_distribucion_produccion.png
│   ├── grafico_2_area_produccion.png
│   ├── grafico_3_produccion_por_anio.png
│   ├── perceptron_curva_aprendizaje.png
│   └── perceptron_frontera_decision.png
├── src/
│   ├── limpiar_datos.py
│   ├── analizar_datos.py
│   ├── graficos_seaborn.py
│   ├── perceptron_proyecto.py
│   ├── entrenar_modelo.py
│   ├── predecir.py
│   ├── graficar_produccion.py
│   └── graficar_prediccion.py
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── requirements.txt
└── README.md
```

## Instalación y ejecución local

Desde la raíz del proyecto, en Linux o WSL:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Ejecutar la preparación, las gráficas y el perceptrón:

```bash
python src/limpiar_datos.py
python src/graficos_seaborn.py
python src/perceptron_proyecto.py
```

Para el modelo principal de regresión:

```bash
python src/entrenar_modelo.py
python src/predecir.py
```

## Ejecución con Docker

Desde la raíz del proyecto, construir la imagen e iniciar la terminal interactiva:

```bash
docker compose run --build --rm prediccion-cafetera
```

El servicio abre la terminal en `/app/src` (el prompt tiene una forma similar a `root@...:/app/src#`). Allí se ejecutan:

```bash
python perceptron_proyecto.py
python graficos_seaborn.py
```

Para ejecutar la limpieza, volver a `/app` porque `limpiar_datos.py` usa rutas relativas:

```bash
cd /app
python src/limpiar_datos.py
```

El volumen configurado en `docker-compose.yml` monta el proyecto en `/app`; por eso, los archivos generados en `data/` permanecen disponibles en el equipo anfitrión.

## Tecnologías

Python, Pandas, NumPy, Seaborn, Matplotlib, Scikit-Learn, Docker, Git y GitHub. Las dependencias de Python se declaran en `requirements.txt`.

## Próximos pasos

- Comparar el modelo de regresión con otros algoritmos de Scikit-Learn y evaluar métricas de regresión.
- Incorporar variables agrícolas adicionales, si se dispone de datos confiables.
- Realizar una evaluación temporal independiente para estimar mejor el desempeño en años futuros.
- Preparar la demostración final con escenarios nuevos para Cartago.
