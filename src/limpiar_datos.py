from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]  # funciona desde cualquier carpeta

# Archivo original
archivo_entrada = RAIZ / "data" / "eva_base_agricola.csv"

# Archivo limpio
archivo_salida = RAIZ / "data" / "cafe_valle_limpio.csv"

# Leer datos
df = pd.read_csv(archivo_entrada)

print("Dimensiones:", df.shape)
print("\nInformación del dataset:")
df.info()

print("\nNulos por columna:")
print(df.isna().sum())

print("\nFilas duplicadas:", df.duplicated().sum())

# Filtrar únicamente Café del Valle del Cauca
cafe = df[
    (df["departamento"].str.upper() == "VALLE DEL CAUCA")
    & (df["cultivo"].str.upper() == "CAFÉ")
].copy()

print("\n=== Diagnóstico de Café del Valle ===")
print("Dimensiones:", cafe.shape)
print("\nNulos por columna:")
print(cafe.isna().sum())
print("\nFilas duplicadas:", cafe.duplicated().sum())

columnas_medidas = [
    "rea_sembrada",
    "rea_cosechada",
    "producci_n",
    "rendimiento",
]

print("\nMínimos de las medidas:")
print(cafe[columnas_medidas].min())

print("\nCantidad de valores negativos:")
print((cafe[columnas_medidas] < 0).sum())

print("\nCultivos encontrados:")
print(cafe["cultivo"].value_counts())

print("\nDepartamentos encontrados:")
print(cafe["departamento"].value_counts())

print("\nCosechadas mayores que sembradas:")
print((cafe["rea_cosechada"] > cafe["rea_sembrada"]).sum())

print("\nPosibles valores atípicos por IQR:")
for columna in columnas_medidas:
    q1 = cafe[columna].quantile(0.25)
    q3 = cafe[columna].quantile(0.75)
    iqr = q3 - q1
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr
    atipicos = cafe[
        (cafe[columna] < limite_inferior)
        | (cafe[columna] > limite_superior)
    ]
    print(f"{columna}: {len(atipicos)}")


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

# Revisar y tratar registros incompletos o medidas imposibles
filas_antes = len(cafe)
nulos_antes = int(cafe.isna().sum().sum())
duplicados_antes = int(cafe.duplicated().sum())

medidas_limpias = [
    "hectareas_sembradas",
    "hectareas_cosechadas",
    "produccion",
    "rendimiento",
]

imposibles = (
    (cafe[medidas_limpias] < 0).any(axis=1)
    | (cafe["hectareas_cosechadas"] > cafe["hectareas_sembradas"])
)

print("\nFilas con medidas imposibles:", int(imposibles.sum()))

cafe = cafe.loc[~imposibles].copy()
cafe = cafe.dropna().drop_duplicates()

print("Filas antes del tratamiento:", filas_antes)
print("Nulos antes del tratamiento:", nulos_antes)
print("Duplicados antes del tratamiento:", duplicados_antes)
print("Filas después del tratamiento:", len(cafe))
print("Nulos después del tratamiento:", int(cafe.isna().sum().sum()))

# Ordenar los datos
cafe = cafe.sort_values(
    ["municipio", "año"]
).reset_index(drop=True)

# Guardar
cafe.to_csv(archivo_salida, index=False, encoding="utf-8")
resumen = pd.DataFrame([
    {
        "estado": "Base EVA original",
        "filas": len(df),
        "columnas": df.shape[1],
        "nulos": int(df.isna().sum().sum()),
    },
    {
        "estado": "Café del Valle limpio",
        "filas": len(cafe),
        "columnas": cafe.shape[1],
        "nulos": int(cafe.isna().sum().sum()),
    },
])

resumen.to_csv(RAIZ / "data" / "resumen_antes_despues.csv", index=False)
print("\nTabla antes/después:")
print(resumen.to_string(index=False))

print("Limpieza completada.")
print(f"Registros guardados: {len(cafe)}")
print(f"Municipios: {cafe['municipio'].nunique()}")
print(f"Años: {cafe['año'].min()} - {cafe['año'].max()}")
print(f"Archivo generado: {archivo_salida}")

print("\nPrimeros registros:")
print(cafe.head().to_string(index=False))
