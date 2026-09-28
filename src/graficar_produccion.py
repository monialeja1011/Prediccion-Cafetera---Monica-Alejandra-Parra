import pandas as pd
import matplotlib.pyplot as plt

archivo = "data/cafe_valle_limpio.csv"

# Leer datos
df = pd.read_csv(archivo)

# Filtrar únicamente Cartago
cartago = df[df["codigo_municipio"] == 76147].copy()

# Ordenar por año
cartago = cartago.sort_values("año")

# Crear gráfica
plt.figure(figsize=(10, 6))

plt.plot(
    cartago["año"],
    cartago["produccion"],
    marker="o",
    linewidth=2
)

plt.title("Producción de Café en Cartago, Valle del Cauca")
plt.xlabel("Año")
plt.ylabel("Producción")
plt.grid(True, alpha=0.3)

# Mostrar el valor sobre cada punto
for _, fila in cartago.iterrows():
    plt.annotate(
        f"{fila['produccion']:.2f}",
        (fila["año"], fila["produccion"]),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center"
    )

plt.tight_layout()

# Guardar gráfica
salida = "data/produccion_cartago.png"
plt.savefig(salida, dpi=300)

print(f"Gráfica guardada en: {salida}")

plt.show()