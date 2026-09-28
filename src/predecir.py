import pandas as pd
import joblib


# Cargar el modelo entrenado
archivo_modelo = "data/modelo_cafetero.pkl"

modelo = joblib.load(archivo_modelo)


# Datos del escenario que queremos predecir
año = 2026
hectareas_sembradas = 280
hectareas_cosechadas = 240


# Crear los datos de entrada
datos = pd.DataFrame(
    {
        "año": [año],
        "hectareas_sembradas": [hectareas_sembradas],
        "hectareas_cosechadas": [hectareas_cosechadas],
    }
)


# Realizar la predicción
prediccion = modelo.predict(datos)


# Mostrar resultado
print("=" * 60)
print("PREDICCIÓN DE PRODUCCIÓN DE CAFÉ")
print("=" * 60)

print("\nMunicipio: Cartago, Valle del Cauca")
print(f"Año proyectado: {año}")
print(f"Hectáreas sembradas: {hectareas_sembradas}")
print(f"Hectáreas cosechadas: {hectareas_cosechadas}")

print(f"\nProducción estimada: {prediccion[0]:.2f}")


print("\nPredicción realizada correctamente.")
