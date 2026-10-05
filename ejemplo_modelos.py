"""Ejemplo de uso del framework: del archivo de datos a la evaluación de los modelos y las gráficas del reporte.

Ejecutar desde la raíz del proyecto:
    PYTHONPATH=src python ejemplo_modelos.py
"""

import pandas as pd

from data_loader import DataLoader
from data_validator import DataValidator
from data_visualizer import ReportFigures
from model_evaluator import ModelEvaluator
from model_trainer import ModelTrainer
from preprocessor import Preprocessor

loader = DataLoader()
validator = DataValidator()
preprocessor = Preprocessor()

fuentes = {
    "perfil_solicitante": ("Docs/perfil_solicitante.csv", validator.limpiar_perfil_solicitante),
    "historial_transaccional": (
        "Docs/historial_transaccional.json",
        validator.limpiar_historial_transaccional,
    ),
    "buro_credito": ("Docs/buro_credito.xlsx", validator.limpiar_buro_credito),
}
calidad, limpios = {}, {}
for fuente, (ruta, limpiar) in fuentes.items():
    original = loader.load(ruta)
    limpios[fuente] = limpiar(original)
    calidad[fuente] = (
        validator.revisar_calidad(original, fuente),
        validator.revisar_calidad(limpios[fuente], fuente),
    )
df_perfil = limpios["perfil_solicitante"]
df_historial = limpios["historial_transaccional"]
df_buro = limpios["buro_credito"]

df_combinado = preprocessor.merge_sources(df_perfil, df_historial, df_buro)
df_etiquetado = preprocessor.define_credit_approval(df_combinado)
datos = preprocessor.prepare_for_model(df_etiquetado)

resultado = ModelTrainer().train(datos)

print("Modelos entrenados:")
print(resultado.resumen().to_string(index=False))

predicciones = pd.DataFrame({"ID_Cliente": datos.ids_test, "real": datos.y_test})
for nombre, modelo in resultado.modelos.items():
    predicciones[f"{nombre} (pred)"] = modelo.predict(datos.X_test)
    predicciones[f"{nombre} (prob)"] = modelo.predict_proba(datos.X_test).round(2)
print("\nPredicciones sobre el conjunto de prueba:")
print(predicciones.to_string(index=False))

print("\nClientes por segmento (K-means):")
print(resultado.segmentos.value_counts().sort_index())

evaluacion = ModelEvaluator().evaluate(resultado)

print("\n=== Evaluación ===")
print("\nComparación de modelos (prueba y validación cruzada):")
print(evaluacion.comparacion.round(3).T.to_string())

recomendacion = evaluacion.recomendacion
print(f"\nModelo recomendado: {recomendacion.modelo} (criterio: {recomendacion.criterio}, "
      f"diferencia significativa: {recomendacion.diferencia_significativa})")
print(recomendacion.justificacion)

for nombre, tabla in evaluacion.importancias.items():
    print(f"\nImportancia de variables — {nombre}:")
    print(tabla.round(3).to_string(index=False))

print("\nSilhouette por número de segmentos:")
print(evaluacion.silhouette.round(3).to_string(index=False))
print(f"Número de segmentos sugerido: {evaluacion.k_sugerido} "
      f"(silhouette del modelo: {evaluacion.silhouette_modelo:.3f})")

print("\nPerfil de cada segmento (escala original):")
print(evaluacion.perfiles_segmentos.round(2).to_string())

graficas = ReportFigures.generar(
    calidad, preprocessor.aggregate_by_client(df_etiquetado), resultado, evaluacion
)
print("\nGráficas guardadas:")
for grafica in ReportFigures.guardar_todas(graficas, "reporte/figuras"):
    print(f"- {grafica.ruta}: {grafica.interpretacion}")
