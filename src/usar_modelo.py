import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


RAIZ = Path(__file__).resolve().parents[1]
ARCHIVO_MODELO = RAIZ / "src" / "modelo_final.joblib"
ARCHIVO_DATOS = RAIZ / "data" / "cafe_valle_limpio.csv"
ARCHIVO_PERCEPTRON = RAIZ / "src" / "perceptron_pesos.npz"

if not ARCHIVO_MODELO.exists():
    sys.exit(
        "No se encontró src/modelo_final.joblib. "
        "Genéralo ejecutando: python src/modelo_proyecto.py"
    )

paquete = joblib.load(ARCHIVO_MODELO)
pipeline = paquete["pipeline"]
caracteristicas = paquete["caracteristicas"]

anios = pd.read_csv(ARCHIVO_DATOS)["año"]
ANIO_MIN = int(anios.min())
ANIO_MAX = int(anios.max())


def probabilidad_sobre_mediana(anio, sembradas):
    if not ARCHIVO_PERCEPTRON.exists():
        return None

    parametros = np.load(ARCHIVO_PERCEPTRON)
    entrada = (np.array([anio, sembradas], dtype=float) - parametros["media"])
    entrada = entrada / parametros["desviacion"]

    z = float(entrada @ parametros["pesos"] + parametros["sesgo"])
    probabilidad = 1 / (1 + np.exp(-z))

    return probabilidad, float(parametros["umbral"])


def estimar(anio, sembradas, cosechadas, etiqueta=""):
    if cosechadas > sembradas:
        print(
            f"[{etiqueta}] Caso inválido: "
            "no se puede cosechar más de lo sembrado."
        )
        return

    caso = pd.DataFrame(
        [[anio, sembradas, cosechadas]],
        columns=caracteristicas,
    )
    toneladas = pipeline.predict(caso)[0]

    print(f"\n{etiqueta}")
    print(
        f"  Año: {anio} | sembradas: {sembradas} ha "
        f"| cosechadas: {cosechadas} ha"
    )
    print(f"  Producción estimada: {toneladas:,.1f} toneladas")

    resultado = probabilidad_sobre_mediana(anio, sembradas)

    if resultado is not None:
        probabilidad, umbral = resultado
        print(
            f"  Perceptrón: probabilidad de superar "
            f"{umbral:,.0f} t (mediana) = {probabilidad:.1%}"
        )

    if anio < ANIO_MIN or anio > ANIO_MAX:
        print(
            f"  Aviso: el modelo se entrenó con años "
            f"{ANIO_MIN}-{ANIO_MAX}; un año fuera de ese rango "
            "es una extrapolación poco fiable."
        )


print("=" * 60)
print(f"ESTIMACIÓN DE PRODUCCIÓN DE CAFÉ - modelo: {paquete['nombre']}")
print("=" * 60)

if len(sys.argv) == 4:
    estimar(
        int(sys.argv[1]),
        float(sys.argv[2]),
        float(sys.argv[3]),
        "Caso ingresado",
    )
else:
    estimar(2026, 280, 240, "Cartago, escenario de demostración")
    estimar(2025, 1300, 1100, "Municipio mediano")
    estimar(2025, 5000, 4800, "Municipio grande")
    estimar(2025, 100, 150, "Caso inválido (cosechadas > sembradas)")

print(
    "\nLa estimación depende de los datos y de las entradas dadas; "
    "no es una medición oficial."
)