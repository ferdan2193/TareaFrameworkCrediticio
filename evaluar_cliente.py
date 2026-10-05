"""Evalúa clientes nuevos con los modelos del framework.

Entrena los modelos con los datos de `Docs/` (igual que `main.py`) y después
aplica a cada cliente nuevo la misma limpieza, imputación y escalado que se
aprendieron en entrenamiento. Para cada cliente muestra:

- La regla de negocio (`credito_aprobado`) y qué condiciones no cumple.
- La predicción y la probabilidad de aprobación de cada modelo supervisado.
- El segmento K-means al que pertenece y el perfil típico de ese segmento.

Uso, desde la raíz del proyecto:

    # Un cliente con argumentos (una sola transacción):
    python evaluar_cliente.py --edad 35 --ingreso 25000 --antiguedad 4.5 \\
        --vivienda Propia --monto 1200 --saldo 15400 --frecuencia 4 \\
        --score 720 --creditos 2 --morosidad NO

    # Uno o varios clientes desde un JSON (ver clientes_nuevos_ejemplo.json):
    python evaluar_cliente.py --archivo clientes_nuevos_ejemplo.json

Formato del JSON: un objeto o una lista de objetos con las claves `perfil`,
`historial` (lista de transacciones) y `buro`, con las mismas columnas que los
archivos de `Docs/`. `ID_Cliente` es opcional; los datos que falten se tratan
como "ND" y se imputan con los valores de entrenamiento.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Permite importar los componentes de src/ sin configurar PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from data_loader import DataLoader
from data_validator import DataValidator
from model_evaluator import ModelEvaluator
from model_trainer import ModelTrainer
from preprocessor import ModelData, Preprocessor


def entrenar(validator: DataValidator, preprocessor: Preprocessor):
    """Ejecuta el pipeline de `main.py` y devuelve el entrenamiento y la evaluación."""
    loader = DataLoader()
    perfil = validator.limpiar_perfil_solicitante(loader.load("Docs/perfil_solicitante.csv"))
    historial = validator.limpiar_historial_transaccional(
        loader.load("Docs/historial_transaccional.json")
    )
    buro = validator.limpiar_buro_credito(loader.load("Docs/buro_credito.xlsx"))

    etiquetada = preprocessor.define_credit_approval(
        preprocessor.merge_sources(perfil, historial, buro)
    )
    datos = preprocessor.prepare_for_model(etiquetada)
    resultado = ModelTrainer().train(datos)
    evaluacion = ModelEvaluator().evaluate(resultado)
    return resultado, evaluacion


def leer_clientes(args: argparse.Namespace) -> list[dict]:
    """Lee los clientes del JSON o arma uno con los argumentos de la línea de comandos."""
    if args.archivo:
        contenido = json.loads(Path(args.archivo).read_text(encoding="utf-8"))
        clientes = contenido if isinstance(contenido, list) else [contenido]
    else:
        clientes = [{
            "perfil": {
                "Edad": args.edad,
                "Ingreso_Declarado": args.ingreso,
                "Antiguedad_Laboral": args.antiguedad,
                "Tipo_Vivienda": args.vivienda,
            },
            "historial": [{
                "Monto_Transaccion": args.monto,
                "Moneda": args.moneda,
                "Saldo_Promedio_Mensual": args.saldo,
                "Frecuencia_Deposito": args.frecuencia,
            }],
            "buro": {
                "Score_Buro": args.score,
                "Creditos_Activos": args.creditos,
                "Mantiene_Morosidad_Previa": args.morosidad,
            },
        }]

    for numero, cliente in enumerate(clientes, start=1):
        cliente.setdefault("ID_Cliente", f"NUEVO-{numero:03d}")
    return clientes


def tablas_de_fuentes(clientes: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convierte los clientes en las tres fuentes, con las columnas que espera `DataValidator`."""
    filas = {"perfil_solicitante": [], "historial_transaccional": [], "buro_credito": []}
    for cliente in clientes:
        id_cliente = cliente["ID_Cliente"]
        filas["perfil_solicitante"].append({**cliente.get("perfil", {}), "ID_Cliente": id_cliente})
        filas["buro_credito"].append({**cliente.get("buro", {}), "ID_Cliente": id_cliente})
        for transaccion in cliente.get("historial", []):
            filas["historial_transaccional"].append({**transaccion, "ID_Cliente": id_cliente})

    tablas = []
    for fuente, registros in filas.items():
        columnas = DataValidator.COLUMNAS_ESPERADAS[fuente]
        tablas.append(pd.DataFrame(registros).reindex(columns=columnas))
    return tuple(tablas)


def transformar(por_cliente: pd.DataFrame, datos: ModelData) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Imputa, codifica y escala con los parámetros aprendidos en entrenamiento.

    Returns:
        `(X, imputados)`: las variables listas para los modelos, con las
        columnas de `datos.variables`, y una tabla booleana que indica qué
        valores faltaban y se imputaron.
    """
    X = por_cliente.set_index("ID_Cliente").drop(columns=["credito_aprobado"])
    X = X.drop(columns=[c for c in datos.descartadas if c in X.columns])
    imputados = X.isna()
    X = X.fillna(datos.valores_imputacion).infer_objects()

    # Columnas de texto: mismas columnas 0/1 que en entrenamiento (categorías nuevas quedan en 0).
    texto = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    X = pd.get_dummies(X, columns=texto, dtype=int)

    for columna, (media, desviacion) in datos.parametros_escalado.items():
        X[columna] = (X[columna] - media) / desviacion
    return X.reindex(columns=datos.variables, fill_value=0).astype(float), imputados


def motivos_rechazo(fila: pd.Series) -> list[str]:
    """Condiciones de la regla de negocio que el cliente no cumple."""
    p = Preprocessor
    motivos = []
    if not p.EDAD_MINIMA_APROBACION <= fila["Edad"] <= p.EDAD_MAXIMA_APROBACION:
        motivos.append(
            f"edad fuera de {p.EDAD_MINIMA_APROBACION}-{p.EDAD_MAXIMA_APROBACION} o no válida"
        )
    if not fila["Score_Buro"] > p.SCORE_MINIMO_APROBACION:
        motivos.append(f"score de buró no mayor a {p.SCORE_MINIMO_APROBACION} o faltante")
    if fila["Mantiene_Morosidad_Previa"] != 0:
        motivos.append("tiene morosidad previa")
    if not fila["Frecuencia_Deposito_Max"] > 0:
        motivos.append("sin depósitos registrados")
    return motivos


def evaluar(clientes: list[dict]) -> pd.DataFrame:
    """Evalúa a los clientes nuevos e imprime el detalle de cada uno."""
    validator = DataValidator()
    preprocessor = Preprocessor()

    print("Entrenando los modelos con los datos de Docs/ ...")
    resultado, evaluacion = entrenar(validator, preprocessor)
    datos = resultado.datos
    recomendado = evaluacion.recomendacion.modelo
    print(f"Listo. Modelo recomendado por la evaluación: {recomendado}")

    perfil, historial, buro = tablas_de_fuentes(clientes)
    combinada = preprocessor.merge_sources(
        validator.limpiar_perfil_solicitante(perfil),
        validator.limpiar_historial_transaccional(historial),
        validator.limpiar_buro_credito(buro),
    )
    por_cliente = preprocessor.aggregate_by_client(preprocessor.define_credit_approval(combinada))
    X, imputados = transformar(por_cliente, datos)
    original = por_cliente.set_index("ID_Cliente")

    predicciones = {
        nombre: (modelo.predict(X), modelo.predict_proba(X))
        for nombre, modelo in resultado.modelos.items()
    }
    segmentos = resultado.segmentacion.predict(X)
    perfiles = evaluacion.perfiles_segmentos

    filas = []
    for id_cliente in X.index:
        fila = original.loc[id_cliente]
        regla = int(fila["credito_aprobado"])
        print(f"\n{'=' * 70}\nCliente {id_cliente}\n{'=' * 70}")
        print(fila.drop("credito_aprobado").to_string())

        faltantes = imputados.loc[id_cliente]
        faltantes = [c for c in faltantes.index[faltantes] if c in datos.valores_imputacion]
        if faltantes:
            print("\nDatos faltantes o no válidos, imputados con valores de entrenamiento:")
            for columna in faltantes:
                print(f"  - {columna} = {datos.valores_imputacion[columna]}")

        print(f"\nRegla de negocio: {'APROBADO' if regla else 'NO APROBADO'}")
        for motivo in motivos_rechazo(fila) if not regla else []:
            print(f"  - {motivo}")

        print("\nModelos:")
        resumen = {"ID_Cliente": id_cliente, "regla_negocio": regla}
        for nombre, (prediccion, probabilidad) in predicciones.items():
            marca = "  <- recomendado" if nombre == recomendado else ""
            print(f"  {nombre:<22} {'APROBADO' if prediccion[id_cliente] else 'NO APROBADO':<12} "
                  f"probabilidad de aprobación {probabilidad[id_cliente]:.2f}{marca}")
            resumen[f"{nombre} (pred)"] = int(prediccion[id_cliente])
            resumen[f"{nombre} (prob)"] = round(float(probabilidad[id_cliente]), 2)

        segmento = int(segmentos[id_cliente])
        resumen["segmento"] = segmento
        print(f"\nSegmento K-means: {segmento}")
        if segmento in perfiles.index:
            print("Perfil típico del segmento:")
            print(perfiles.loc[segmento].round(2).to_string())
        filas.append(resumen)

    tabla = pd.DataFrame(filas)
    print(f"\n{'=' * 70}\nResumen\n{'=' * 70}")
    print(tabla.to_string(index=False))
    return tabla


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa clientes nuevos con los modelos del framework.")
    parser.add_argument("--archivo", help="JSON con uno o varios clientes")
    parser.add_argument("--salida", help="CSV donde guardar el resumen (opcional)")
    perfil = parser.add_argument_group("perfil del solicitante")
    perfil.add_argument("--edad", type=float)
    perfil.add_argument("--ingreso", type=float, help="Ingreso declarado mensual")
    perfil.add_argument("--antiguedad", type=float, help="Antigüedad laboral en años")
    perfil.add_argument("--vivienda", help="Propia, Rentada, Hipotecada, ...")
    historial = parser.add_argument_group("historial transaccional (una transacción)")
    historial.add_argument("--monto", type=float, help="Monto de la transacción")
    historial.add_argument("--moneda", default="MXN", help="MXN o USD (default: MXN)")
    historial.add_argument("--saldo", type=float, help="Saldo promedio mensual")
    historial.add_argument("--frecuencia", type=float, help="Frecuencia de depósito")
    buro = parser.add_argument_group("buró de crédito")
    buro.add_argument("--score", type=float, help="Score de buró")
    buro.add_argument("--creditos", type=float, help="Créditos activos")
    buro.add_argument("--morosidad", default="NO", help="SI / NO (default: NO)")
    args = parser.parse_args()

    tabla = evaluar(leer_clientes(args))
    if args.salida:
        tabla.to_csv(args.salida, index=False)
        print(f"\nResumen guardado en {args.salida}")


if __name__ == "__main__":
    main()
