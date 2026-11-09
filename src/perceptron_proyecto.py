from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

RAIZ = Path(__file__).resolve().parents[1]
df = pd.read_csv(RAIZ / "data" / "cafe_valle_limpio.csv")

COLUMNAS = ["año", "hectareas_sembradas"]

rng = np.random.default_rng(42)
indices = rng.permutation(len(df))
corte = int(0.8 * len(df))
idx_ent, idx_prueba = indices[:corte], indices[corte:]

df_ent, df_prueba = df.iloc[idx_ent], df.iloc[idx_prueba]

umbral = float(df_ent["produccion"].median())
y_ent = (df_ent["produccion"] > umbral).astype(float).to_numpy()
y_prueba = (df_prueba["produccion"] > umbral).astype(int).to_numpy()

X_ent_orig = df_ent[COLUMNAS].to_numpy(dtype=float)
X_prueba_orig = df_prueba[COLUMNAS].to_numpy(dtype=float)
media = X_ent_orig.mean(axis=0)
desviacion = X_ent_orig.std(axis=0)
X_ent = (X_ent_orig - media) / desviacion
X_prueba = (X_prueba_orig - media) / desviacion

print(f"Umbral (mediana en entrenamiento): {umbral:.2f} toneladas")
print(f"Entrenamiento: {len(X_ent)} | Prueba: {len(X_prueba)}")


def sigmoide(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def perdida_logaritmica(y_real, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -np.mean(y_real * np.log(p) + (1 - y_real) * np.log(1 - p))


pesos = np.zeros(X_ent.shape[1])
sesgo = 0.0
tasa_aprendizaje = 0.1
epocas = 5000
historial = []

for _ in range(epocas):
    p = sigmoide(X_ent @ pesos + sesgo)
    error = p - y_ent
    pesos -= tasa_aprendizaje * (X_ent.T @ error) / len(y_ent)
    sesgo -= tasa_aprendizaje * error.mean()
    historial.append(perdida_logaritmica(y_ent, sigmoide(X_ent @ pesos + sesgo)))


def predecir(X):
    return (sigmoide(X @ pesos + sesgo) >= 0.5).astype(int)


exac_ent = np.mean(predecir(X_ent) == y_ent)
pred_prueba = predecir(X_prueba)
exac_prueba = np.mean(pred_prueba == y_prueba)

matriz = np.zeros((2, 2), dtype=int)
for real, pred in zip(y_prueba, pred_prueba):
    matriz[real, pred] += 1

print("\n=== Resultados del perceptrón ===")
print(f"w1 (año): {pesos[0]:.4f}")
print(f"w2 (hectáreas sembradas): {pesos[1]:.4f}")
print(f"Sesgo (b): {sesgo:.4f}")
print(f"Pérdida inicial: {np.log(2):.4f} | final: {historial[-1]:.4f}")
print(f"Exactitud en entrenamiento: {exac_ent:.4f}")
print(f"Exactitud en prueba:        {exac_prueba:.4f}")
print("Matriz de confusión (prueba) [real x predicho]:")
print(matriz)

np.savez(
    RAIZ / "src" / "perceptron_pesos.npz",
    pesos=pesos, sesgo=sesgo, media=media, desviacion=desviacion, umbral=umbral,
)

plt.figure(figsize=(9, 5))
plt.plot(range(1, epocas + 1), historial)
plt.title("Curva de aprendizaje del perceptrón")
plt.xlabel("Época")
plt.ylabel("Pérdida logarítmica")
plt.tight_layout()
plt.savefig(RAIZ / "data" / "perceptron_curva_aprendizaje.png", dpi=300)
plt.close()

plt.figure(figsize=(9, 6))
sns.scatterplot(
    x=X_ent[:, 0], y=X_ent[:, 1],
    hue=np.where(y_ent == 1, "Producción > mediana", "Producción ≤ mediana"),
    alpha=0.8,
)
x_linea = np.linspace(X_ent[:, 0].min(), X_ent[:, 0].max(), 200)
plt.plot(x_linea, -(pesos[0] * x_linea + sesgo) / pesos[1],
         "k--", label="Frontera p = 0.5")
plt.title("Frontera de decisión del perceptrón (datos de entrenamiento)")
plt.xlabel("Año (estandarizado)")
plt.ylabel("Hectáreas sembradas (estandarizadas)")
plt.legend()
plt.tight_layout()
plt.savefig(RAIZ / "data" / "perceptron_frontera_decision.png", dpi=300)
plt.close()

plt.figure(figsize=(5, 4))
sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["≤ mediana", "> mediana"],
            yticklabels=["≤ mediana", "> mediana"])
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.title("Matriz de confusión del perceptrón (prueba)")
plt.tight_layout()
plt.savefig(RAIZ / "data" / "perceptron_matriz_confusion.png", dpi=300)
plt.close()
print("Gráficos guardados en data/")