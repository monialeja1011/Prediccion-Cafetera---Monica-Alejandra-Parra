from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RAIZ = Path(__file__).resolve().parents[1]
archivo = RAIZ / "data" / "cafe_valle_limpio.csv"
df = pd.read_csv(archivo)

# Objetivo binario: 1 si la producción supera la mediana; 0 si no.
umbral = float(df["produccion"].median())
y = (df["produccion"] > umbral).astype(int).to_numpy()

# Dos entradas numéricas.
columnas_entrada = ["año", "hectareas_sembradas"]
X_original = df[columnas_entrada].to_numpy(dtype=float)

# Estandarización: (valor - media) / desviación estándar.
media = X_original.mean(axis=0)
desviacion = X_original.std(axis=0)
X = (X_original - media) / desviacion

print("Umbral de producción:", umbral)
print("Cantidad clase 0:", int((y == 0).sum()))
print("Cantidad clase 1:", int((y == 1).sum()))
print("Entradas:", columnas_entrada)
print("Media de las entradas:", media)
print("Desviación estándar:", desviacion)
print("Dimensiones de X:", X.shape)
def sigmoide(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def perdida_logaritmica(y_real, probabilidades):
    epsilon = 1e-12
    probabilidades = np.clip(
        probabilidades,
        epsilon,
        1 - epsilon,
    )
    return -np.mean(
        y_real * np.log(probabilidades)
        + (1 - y_real) * np.log(1 - probabilidades)
    )


# Inicializar parámetros
pesos = np.zeros(X.shape[1])
sesgo = 0.0
tasa_aprendizaje = 0.1
epocas = 5000
historial_perdida = []

y_entrenamiento = y.astype(float)

perdida_inicial = perdida_logaritmica(
    y_entrenamiento,
    sigmoide(X @ pesos + sesgo),
)

# Descenso del gradiente
for epoca in range(epocas):
    probabilidades = sigmoide(X @ pesos + sesgo)
    error = probabilidades - y_entrenamiento

    gradiente_pesos = (X.T @ error) / len(y_entrenamiento)
    gradiente_sesgo = error.mean()

    pesos -= tasa_aprendizaje * gradiente_pesos
    sesgo -= tasa_aprendizaje * gradiente_sesgo

    perdida_actual = perdida_logaritmica(
        y_entrenamiento,
        sigmoide(X @ pesos + sesgo),
    )
    historial_perdida.append(perdida_actual)

# Evaluación sobre los datos usados para entrenar
probabilidades_finales = sigmoide(X @ pesos + sesgo)
predicciones = (probabilidades_finales >= 0.5).astype(int)
exactitud = np.mean(predicciones == y)

print("\n=== Resultados del perceptrón ===")
print(f"w1 (año): {pesos[0]:.6f}")
print(f"w2 (hectáreas sembradas): {pesos[1]:.6f}")
print(f"Sesgo (b): {sesgo:.6f}")
print(f"Pérdida inicial: {perdida_inicial:.6f}")
print(f"Pérdida final: {historial_perdida[-1]:.6f}")
print(f"Exactitud de entrenamiento: {exactitud:.4f}")
# Gráfica 1: curva de aprendizaje
plt.figure(figsize=(9, 5))
plt.plot(range(1, epocas + 1), historial_perdida)

plt.title("Curva de aprendizaje del perceptrón")
plt.xlabel("Época")
plt.ylabel("Pérdida logarítmica")
plt.tight_layout()

salida = RAIZ / "data" / "perceptron_curva_aprendizaje.png"
plt.savefig(salida, dpi=300)
plt.close()
print(f"Gráfica guardada en: {salida}")


# Gráfica 2: frontera de decisión
datos_grafica = pd.DataFrame({
    "Año estandarizado": X[:, 0],
    "Hectáreas sembradas estandarizadas": X[:, 1],
    "Clase": np.where(y == 1, "Producción > mediana", "Producción ≤ mediana"),
})

plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=datos_grafica,
    x="Año estandarizado",
    y="Hectáreas sembradas estandarizadas",
    hue="Clase",
    alpha=0.8,
)

x_linea = np.linspace(X[:, 0].min(), X[:, 0].max(), 200)
y_linea = -(pesos[0] * x_linea + sesgo) / pesos[1]
plt.plot(x_linea, y_linea, color="black", linestyle="--", label="Frontera p = 0.5")

plt.title("Frontera de decisión del perceptrón")
plt.xlabel("Año (estandarizado)")
plt.ylabel("Hectáreas sembradas (estandarizadas)")
plt.legend()
plt.tight_layout()

salida = RAIZ / "data" / "perceptron_frontera_decision.png"
plt.savefig(salida, dpi=300)
plt.close()
print(f"Gráfica guardada en: {salida}")