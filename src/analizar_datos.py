import pandas as pd

archivo = "data/cafe_valle_limpio.csv"

df = pd.read_csv(archivo)

print("=" * 60)
print("ANÁLISIS DE PRODUCCIÓN DE CAFÉ")
print("=" * 60)

print(f"\nTotal de registros: {len(df)}")
print(f"Municipios: {df['municipio'].nunique()}")
print(f"Años: {df['año'].min()} - {df['año'].max()}")

print("\nMunicipios:")
print(df["municipio"].unique())

print("\nEstadísticas descriptivas:")
print(
    df[
        [
            "hectareas_sembradas",
            "hectareas_cosechadas",
            "produccion",
        ]
    ].describe()
)

print("\nProducción promedio por año:")
print(
    df.groupby("año")["produccion"]
    .mean()
    .round(2)
)

print("\nProducción de Cartago:")
cartago = df[df["codigo_municipio"] == 76147]

print(
    cartago[
        [
            "año",
            "hectareas_sembradas",
            "hectareas_cosechadas",
            "produccion",
        ]
    ].to_string(index=False)
)

print("\nAnálisis terminado.")
