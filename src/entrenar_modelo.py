import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Archivo con los datos limpios
archivo = "data/cafe_valle_limpio.csv"

# Leer datos
df = pd.read_csv(archivo)


# Variables que utilizará el modelo
caracteristicas = [
    "año",
    "hectareas_sembradas",
    "hectareas_cosechadas",
]

# Variable que queremos predecir
objetivo = "produccion"


X = df[caracteristicas]
y = df[objetivo]


# Dividir los datos:
# 80% para entrenamiento
# 20% para prueba
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# Crear el modelo
modelo = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)


# Entrenar el modelo
modelo.fit(X_entrenamiento, y_entrenamiento)


# Realizar predicciones sobre los datos de prueba
predicciones = modelo.predict(X_prueba)


# Evaluar el modelo
mae = mean_absolute_error(y_prueba, predicciones)
rmse = mean_squared_error(y_prueba, predicciones) ** 0.5
r2 = r2_score(y_prueba, predicciones)


print("=" * 60)
print("MODELO DE PREDICCIÓN DE PRODUCCIÓN DE CAFÉ")
print("=" * 60)

print(f"\nTotal de registros: {len(df)}")
print(f"Registros para entrenamiento: {len(X_entrenamiento)}")
print(f"Registros para prueba: {len(X_prueba)}")

print("\nVariables utilizadas:")
for variable in caracteristicas:
    print(f"- {variable}")

print(f"\nVariable objetivo: {objetivo}")

print("\nResultados del modelo:")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.4f}")


# Guardar el modelo
archivo_modelo = "data/modelo_cafetero.pkl"

joblib.dump(modelo, archivo_modelo)

print(f"\nModelo guardado en: {archivo_modelo}")