import pandas as pd

# Archivo original
archivo_entrada = "data/eva_base_agricola.csv"

# Archivo limpio
archivo_salida = "data/cafe_valle_limpio.csv"

# Leer datos
df = pd.read_csv(archivo_entrada)

# Filtrar únicamente Café del Valle del Cauca
cafe = df[
    (df["departamento"].str.upper() == "VALLE DEL CAUCA")
    & (df["cultivo"].str.upper() == "CAFÉ")
].copy()

# Seleccionar las variables principales
cafe = cafe[
    [
        "c_digo_dane_municipio",
        "municipio",
        "a_o",
        "rea_sembrada",
        "rea_cosechada",
        "producci_n",
        "rendimiento",
    ]
]

# Renombrar las columnas para que sean fáciles de utilizar
cafe = cafe.rename(
    columns={
        "c_digo_dane_municipio": "codigo_municipio",
        "municipio": "municipio",
        "a_o": "año",
        "rea_sembrada": "hectareas_sembradas",
        "rea_cosechada": "hectareas_cosechadas",
        "producci_n": "produccion",
        "rendimiento": "rendimiento",
    }
)

# Convertir variables numéricas
columnas_numericas = [
    "codigo_municipio",
    "año",
    "hectareas_sembradas",
    "hectareas_cosechadas",
    "produccion",
    "rendimiento",
]

for columna in columnas_numericas:
    cafe[columna] = pd.to_numeric(cafe[columna], errors="coerce")

# Eliminar registros incompletos
cafe = cafe.dropna()

# Ordenar los datos
cafe = cafe.sort_values(
    ["municipio", "año"]
).reset_index(drop=True)

# Guardar
cafe.to_csv(archivo_salida, index=False, encoding="utf-8")

print("Limpieza completada.")
print(f"Registros guardados: {len(cafe)}")
print(f"Municipios: {cafe['municipio'].nunique()}")
print(f"Años: {cafe['año'].min()} - {cafe['año'].max()}")
print(f"Archivo generado: {archivo_salida}")

print("\nPrimeros registros:")
print(cafe.head().to_string(index=False))
