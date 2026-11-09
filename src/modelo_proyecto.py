from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RAIZ = Path(__file__).resolve().parents[1]
ARCHIVO_DATOS = RAIZ / "data" / "cafe_valle_limpio.csv"
ARCHIVO_MODELO = RAIZ / "src" / "modelo_final.joblib"
SEMILLA = 42

df = pd.read_csv(ARCHIVO_DATOS)

CARACTERISTICAS = [
    "año",
    "hectareas_sembradas",
    "hectareas_cosechadas",
]
OBJETIVO = "produccion"

X = df[CARACTERISTICAS]
y = df[OBJETIVO]
grupos = df["municipio"]

X_ent, X_prueba, y_ent, y_prueba = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEMILLA,
)

print("=" * 64)
print("COMPARACIÓN DE MODELOS - PRODUCCIÓN DE CAFÉ (toneladas)")
print("=" * 64)
print(
    f"Registros: {len(df)} | "
    f"entrenamiento: {len(X_ent)} | prueba: {len(X_prueba)}"
)
print(f"Entradas: {CARACTERISTICAS}")

modelos = {
    "Regresión lineal": Pipeline([
        ("escalador", StandardScaler()),
        ("modelo", LinearRegression()),
    ]),
    "Random Forest": Pipeline([
        ("escalador", StandardScaler()),
        ("modelo", RandomForestRegressor(
            n_estimators=200,
            random_state=SEMILLA,
        )),
    ]),
    "Gradient Boosting": Pipeline([
        ("escalador", StandardScaler()),
        ("modelo", GradientBoostingRegressor(random_state=SEMILLA)),
    ]),
}

kfold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=SEMILLA,
)
metricas_cv = {
    "MAE": "neg_mean_absolute_error",
    "RMSE": "neg_root_mean_squared_error",
    "R2": "r2",
}

filas = []

for nombre, pipeline in modelos.items():
    resultados_cv = cross_validate(
        pipeline,
        X_ent,
        y_ent,
        cv=kfold,
        scoring=metricas_cv,
    )

    pipeline.fit(X_ent, y_ent)
    predicciones = pipeline.predict(X_prueba)

    filas.append({
        "Modelo": nombre,
        "MAE (CV)": -resultados_cv["test_MAE"].mean(),
        "RMSE (CV)": -resultados_cv["test_RMSE"].mean(),
        "R2 (CV)": resultados_cv["test_R2"].mean(),
        "MAE (prueba)": mean_absolute_error(y_prueba, predicciones),
        "RMSE (prueba)": mean_squared_error(
            y_prueba,
            predicciones,
        ) ** 0.5,
        "R2 (prueba)": r2_score(y_prueba, predicciones),
    })

resultados = pd.DataFrame(filas).set_index("Modelo")

print("\n--- Resultados (toneladas; CV = promedio de 5 pliegues) ---")
print(resultados.round(3).to_string())

mejor_nombre = resultados["RMSE (CV)"].idxmin()
print(f"\nMejor modelo según RMSE en validación cruzada: {mejor_nombre}")

grupo_cv = GroupKFold(n_splits=5)
filas_agrupadas = []

for nombre, pipeline in modelos.items():
    resultados_cv = cross_validate(
        pipeline,
        X,
        y,
        cv=grupo_cv,
        groups=grupos,
        scoring=metricas_cv,
    )

    filas_agrupadas.append({
        "Modelo": nombre,
        "MAE": -resultados_cv["test_MAE"].mean(),
        "RMSE": -resultados_cv["test_RMSE"].mean(),
        "R2": resultados_cv["test_R2"].mean(),
    })

resultados_agrupados = pd.DataFrame(filas_agrupadas).set_index("Modelo")

print("\n--- Validación cruzada agrupada por municipio ---")
print(resultados_agrupados.round(3).to_string())

modelo_final = modelos[mejor_nombre]
modelo_final.fit(X_ent, y_ent)
predicciones_prueba = modelo_final.predict(X_prueba)

mae_prueba = mean_absolute_error(y_prueba, predicciones_prueba)
rmse_prueba = mean_squared_error(y_prueba, predicciones_prueba) ** 0.5
r2_prueba = r2_score(y_prueba, predicciones_prueba)

print(f"\n--- {mejor_nombre} en el conjunto de prueba ---")
print(f"MAE : {mae_prueba:.2f} toneladas")
print(f"RMSE: {rmse_prueba:.2f} toneladas")
print(f"R²  : {r2_prueba:.4f}")

joblib.dump(
    {
        "nombre": mejor_nombre,
        "caracteristicas": CARACTERISTICAS,
        "pipeline": modelo_final,
    },
    ARCHIVO_MODELO,
)
print(f"\nModelo guardado en: {ARCHIVO_MODELO}")

plt.figure(figsize=(7, 6))
plt.scatter(y_prueba, predicciones_prueba, alpha=0.8)

limite = max(y_prueba.max(), predicciones_prueba.max()) * 1.05
plt.plot(
    [0, limite],
    [0, limite],
    "k--",
    label="Predicción perfecta",
)
plt.xlabel("Producción real (toneladas)")
plt.ylabel("Producción predicha (toneladas)")
plt.title(f"Real vs predicho - {mejor_nombre} (conjunto de prueba)")
plt.legend()
plt.tight_layout()
plt.savefig(RAIZ / "data" / "modelo_real_vs_predicho.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
plt.bar(resultados.index, resultados["RMSE (CV)"])
plt.ylabel("RMSE en validación cruzada (toneladas)")
plt.title("Comparación de modelos (menor es mejor)")
plt.tight_layout()
plt.savefig(RAIZ / "data" / "modelo_comparacion.png", dpi=300)
plt.close()

print(
    "Gráficos guardados en data/: "
    "modelo_real_vs_predicho.png y modelo_comparacion.png"
)