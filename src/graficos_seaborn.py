from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parents[1]
archivo = RAIZ / "data" / "cafe_valle_limpio.csv"
df = pd.read_csv(archivo)

sns.set_theme(style="whitegrid")

# Gráfica 1: distribución de la producción
plt.figure(figsize=(9, 5))
sns.histplot(data=df, x="produccion", bins=25, kde=True)

plt.title("Distribución de la producción de café")
plt.xlabel("Producción (unidad reportada en la fuente)")
plt.ylabel("Número de registros")
plt.tight_layout()

salida = RAIZ / "data" / "grafico_1_distribucion_produccion.png"
plt.savefig(salida, dpi=300)
plt.close()
print(f"Gráfica guardada en: {salida}")

# Gráfica 2: relación entre área cosechada y producción
plt.figure(figsize=(9, 5))
sns.scatterplot(
    data=df,
    x="hectareas_cosechadas",
    y="produccion",
    alpha=0.7,
)

plt.title("Área cosechada y producción de café")
plt.xlabel("Hectáreas cosechadas")
plt.ylabel("Producción (unidad reportada en la fuente)")
plt.tight_layout()

salida = RAIZ / "data" / "grafico_2_area_produccion.png"
plt.savefig(salida, dpi=300)
plt.close()
print(f"Gráfica guardada en: {salida}")

# Gráfica 3: distribución de la producción por año
plt.figure(figsize=(9, 5))
sns.boxplot(
    data=df,
    x="año",
    y="produccion",
)

plt.title("Distribución de la producción de café por año")
plt.xlabel("Año")
plt.ylabel("Producción (unidad reportada en la fuente)")
plt.tight_layout()

salida = RAIZ / "data" / "grafico_3_produccion_por_anio.png"
plt.savefig(salida, dpi=300)
plt.close()
print(f"Gráfica guardada en: {salida}")