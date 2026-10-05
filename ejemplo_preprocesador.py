"""Ejemplo de uso del framework: carga, limpieza, fusión, etiquetado y preparación para el modelo.

Ejecutar desde la raíz del proyecto:
    PYTHONPATH=src python ejemplo_preprocesador.py
"""

from data_loader import DataLoader
from data_validator import DataValidator
from preprocessor import Preprocessor

loader = DataLoader()
validator = DataValidator()
preprocessor = Preprocessor()

df_perfil = validator.limpiar_perfil_solicitante(loader.load("Docs/perfil_solicitante.csv"))
df_historial = validator.limpiar_historial_transaccional(
    loader.load("Docs/historial_transaccional.json")
)
df_buro = validator.limpiar_buro_credito(loader.load("Docs/buro_credito.xlsx"))

df_combinado = preprocessor.merge_sources(df_perfil, df_historial, df_buro)
df_combinado = preprocessor.define_credit_approval(df_combinado)
print(df_combinado)

print("\nClientes por clase de credito_aprobado:")
print(preprocessor.approval_summary(df_combinado))

figura = preprocessor.show_combined_table(df_combinado, title="Tabla combinada")
figura.savefig("tabla_combinada.png")
print("\nImagen guardada en tabla_combinada.png")

datos = preprocessor.prepare_for_model(df_combinado)
print("\nPreparación para el modelo:")
print("X_train:", datos.X_train.shape, "| X_test:", datos.X_test.shape)
print("Variables:", datos.variables)
print("Descartadas:", datos.descartadas)
