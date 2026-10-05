"""Flujo completo del framework de análisis crediticio, paso a paso.

Muestra cómo se llama a cada componente, en el orden de `Docs/diagram.md`:

    DataLoader → DataValidator → Preprocessor → ModelTrainer → ModelEvaluator → DataVisualizer

Ejecutar desde la raíz del proyecto:
    python main.py
"""

import sys
from pathlib import Path

# Permite importar los componentes de src/ sin configurar PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from data_loader import DataLoader
from data_validator import DataValidator
from data_visualizer import ReportFigures
from model_evaluator import ModelEvaluator
from model_trainer import ModelTrainer
from preprocessor import Preprocessor

CARPETA_FIGURAS = "reporte/figuras"


def titulo(numero: int, texto: str) -> None:
    print(f"\n{'=' * 70}\n{numero}. {texto}\n{'=' * 70}")


def main() -> None:
    # Un objeto por componente del pipeline.
    loader = DataLoader()
    validator = DataValidator()          # tasa_cambio_usd_mxn=15 por defecto
    preprocessor = Preprocessor()
    trainer = ModelTrainer()             # random_state=42, n_segmentos=3
    evaluator = ModelEvaluator()         # n_particiones=5, random_state=42

    # ------------------------------------------------------------------
    # 1. CARGA: DataLoader lee CSV, JSON o XLSX y devuelve un DataFrame.
    # ------------------------------------------------------------------
    titulo(1, "Carga de datos (DataLoader)")
    perfil = loader.load("Docs/perfil_solicitante.csv")
    historial = loader.load("Docs/historial_transaccional.json")
    buro = loader.load("Docs/buro_credito.xlsx")
    print(f"Perfil: {perfil.shape} | Historial: {historial.shape} | Buró: {buro.shape}")

    # ------------------------------------------------------------------
    # 2. REVISIÓN DE CALIDAD (antes de limpiar): no modifica los datos.
    # ------------------------------------------------------------------
    titulo(2, "Revisión de estructura y calidad, antes de limpiar")
    calidad_antes = {
        "perfil_solicitante": validator.revisar_calidad(perfil, "perfil_solicitante"),
        "historial_transaccional": validator.revisar_calidad(historial, "historial_transaccional"),
        "buro_credito": validator.revisar_calidad(buro, "buro_credito"),
    }
    for reporte in calidad_antes.values():
        print(reporte.resumen(), "\n")   # texto legible de cada ReporteCalidad

    # ------------------------------------------------------------------
    # 3. LIMPIEZA: DataValidator, un método por fuente.
    #    Normaliza formatos, quita duplicados y marca faltantes/atípicos como "ND".
    # ------------------------------------------------------------------
    titulo(3, "Limpieza (DataValidator)")
    perfil_limpio = validator.limpiar_perfil_solicitante(perfil)
    historial_limpio = validator.limpiar_historial_transaccional(historial)
    buro_limpio = validator.limpiar_buro_credito(buro)

    calidad_despues = {
        "perfil_solicitante": validator.revisar_calidad(perfil_limpio, "perfil_solicitante"),
        "historial_transaccional": validator.revisar_calidad(historial_limpio, "historial_transaccional"),
        "buro_credito": validator.revisar_calidad(buro_limpio, "buro_credito"),
    }
    # Tabla antes/después por fuente. Los faltantes que quedan son "ND".
    for fuente in calidad_antes:
        print(f"\n{fuente}:")
        print(DataValidator.comparar_calidad(calidad_antes[fuente], calidad_despues[fuente]).to_string())

    # ------------------------------------------------------------------
    # 4. COMBINACIÓN: Preprocessor une las 3 fuentes por ID_Cliente.
    # ------------------------------------------------------------------
    titulo(4, "Combinación de fuentes (Preprocessor.merge_sources)")
    combinada = preprocessor.merge_sources(perfil_limpio, historial_limpio, buro_limpio)
    print(f"Tabla combinada: {combinada.shape} ({combinada['ID_Cliente'].nunique()} clientes)")

    # ------------------------------------------------------------------
    # 5. VARIABLE OBJETIVO: regla de negocio de credito_aprobado.
    #    1 si: 18 <= edad <= 69, score > 600, sin morosidad y con depósitos.
    # ------------------------------------------------------------------
    titulo(5, "Variable objetivo (Preprocessor.define_credit_approval)")
    etiquetada = preprocessor.define_credit_approval(combinada)
    resumen = preprocessor.approval_summary(etiquetada)
    print(f"Clientes no aprobados (0): {resumen[0]} | aprobados (1): {resumen[1]}")

    # ------------------------------------------------------------------
    # 6. PREPARACIÓN PARA EL MODELO: una fila por cliente, separar
    #    entrenamiento/prueba, imputar ND, codificar y escalar.
    # ------------------------------------------------------------------
    titulo(6, "Preparación para el modelo (Preprocessor.prepare_for_model)")
    datos = preprocessor.prepare_for_model(etiquetada)
    X_train, X_test, y_train, y_test = datos.as_tuple()
    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"Aprobados en entrenamiento: {int(y_train.sum())} | en prueba: {int(y_test.sum())}")
    print(f"Variables: {datos.variables}")
    print(f"Descartadas: {datos.descartadas}")

    # ------------------------------------------------------------------
    # 7. ENTRENAMIENTO: ModelTrainer entrena 2 modelos supervisados y K-means.
    # ------------------------------------------------------------------
    titulo(7, "Entrenamiento (ModelTrainer)")
    resultado = trainer.train(datos)
    print(resultado.resumen()[["modelo", "tipo", "tiempo_entrenamiento_s"]].to_string(index=False))

    # Ejemplo de uso directo de un modelo entrenado:
    modelo = resultado.modelos["Random forest"]
    print("\nPrimeros 5 clientes de prueba (Random forest):")
    for id_cliente, real, prediccion, probabilidad in zip(
        datos.ids_test[:5], y_test[:5], modelo.predict(X_test)[:5], modelo.predict_proba(X_test)[:5]
    ):
        print(f"  {id_cliente}: real={real} predicción={prediccion} "
              f"probabilidad de aprobación={probabilidad:.2f}")

    # ------------------------------------------------------------------
    # 8. EVALUACIÓN: métricas, validación cruzada, comparación y segmentación.
    # ------------------------------------------------------------------
    titulo(8, "Evaluación (ModelEvaluator)")
    evaluacion = evaluator.evaluate(resultado)
    columnas = ["precision_prueba", "recall_prueba", "f1_prueba", "auc_prueba",
                "exactitud_cv", "precision_cv", "recall_cv", "f1_cv"]
    print(evaluacion.comparacion[columnas].round(3).T.to_string())
    print(f"\nModelo recomendado: {evaluacion.recomendacion.modelo}")
    print(evaluacion.recomendacion.justificacion)
    print(f"\nSegmentación: silhouette {evaluacion.silhouette_modelo:.2f}, "
          f"k sugerido = {evaluacion.k_sugerido}")
    print(evaluacion.perfiles_segmentos[["n_clientes", "Edad", "Score_Buro",
                                         "Mantiene_Morosidad_Previa", "tasa_aprobacion"]]
          .round(2).to_string())

    # ------------------------------------------------------------------
    # 9. VISUALIZACIÓN: 11 gráficas con interpretación, guardadas como PNG.
    # ------------------------------------------------------------------
    titulo(9, "Gráficas (DataVisualizer → ReportFigures)")
    calidad = {fuente: (calidad_antes[fuente], calidad_despues[fuente]) for fuente in calidad_antes}
    graficas = ReportFigures.generar(
        calidad, preprocessor.aggregate_by_client(etiquetada), resultado, evaluacion
    )
    for grafica in ReportFigures.guardar_todas(graficas, CARPETA_FIGURAS):
        print(f"- {grafica.ruta}")

    print(f"\nFlujo completo. Las gráficas y sus interpretaciones están en {CARPETA_FIGURAS}/.")


if __name__ == "__main__":
    main()
