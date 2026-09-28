import pandas as pd
import joblib
import matplotlib.pyplot as plt


# Cargar el modelo entrenado
modelo = joblib.load("data/modelo_cafetero.pkl")


# Datos históricos de Cartago
df = pd.read_csv("data/cafe_valle_limpio.csv")

cartago = df[df["codigo_municipio"] == 76147].copy()
cartago = cartago.sort_values("año")


# Datos para la predicción de 2026
año_prediccion = 2026
hectareas_sembradas = 280
hectareas_cosechadas = 240


datos_prediccion = pd.DataFrame(
    {
        "año": [año_prediccion],
        "hectareas_sembradas": [hectareas_sembradas],
        "hectareas_cosechadas": [hectareas_cosechadas],
    }
)


# Realizar la predicción
prediccion = modelo.predict(datos_prediccion)[0]


# Crear la gráfica
plt.figure(figsize=(10, 6))

plt.plot(
    cartago["año"],
    cartago["produccion"],
    marker="o",
    linewidth=2,
    label="Producción histórica"
)

plt.scatter(
    año_prediccion,
    prediccion,
    s=100,
    label="Predicción 2026"
)

plt.plot(
    [cartago["año"].max(), año_prediccion],
    [cartago["produccion"].iloc[-1], prediccion],
    linestyle="--"
)


# Mostrar valores históricos
for _, fila in cartago.iterrows():
    plt.annotate(
        f"{fila['produccion']:.2f}",
        (fila["año"], fila["produccion"]),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center"
    )


# Mostrar valor de la predicción
plt.annotate(
    f"{prediccion:.2f}",
    (año_prediccion, prediccion),
    textcoords="offset points",
    xytext=(0, 10),
    ha="center"
)


plt.title("Producción de Café en Cartago: Datos Históricos y Predicción")
plt.xlabel("Año")
plt.ylabel("Producción")
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()


# Guardar gráfica
salida = "data/prediccion_cartago.png"

plt.savefig(salida, dpi=300)

print(f"Gráfica guardada en: {salida}")
print(f"Producción estimada para 2026: {prediccion:.2f}")
