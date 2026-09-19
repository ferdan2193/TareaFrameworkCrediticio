# TareaFrameworkCrediticio

Este repositorio incluye un pipeline de datos en tres etapas:

1. Carga de archivos con `DataLoader`.
2. Limpieza y normalización con `DataValidator`.
3. Fusión de fuentes con `Preprocessor`.

## Requisitos

- Python 3.10+
- Dependencias:
  - pandas
  - numpy
  - matplotlib
  - openpyxl

Instalación rápida:

```bash
pip install pandas numpy matplotlib openpyxl
```

## Estructura esperada de datos de entrada

El ejemplo usa estos archivos dentro de `Docs/`:

- `Docs/perfil_solicitante.csv`
- `Docs/historial_transaccional.json`
- `Docs/buro_credito.xlsx`

## Ejecutar el pipeline completo

Desde la raíz del proyecto, ejecuta:

```bash
python ejemplo_pipeline.py
```

El script:

1. Carga las tres fuentes con `DataLoader`.
2. Limpia cada DataFrame con su método de `DataValidator`.
3. Une todo por `ID_Cliente` con `Preprocessor.merge_sources`.
4. Guarda el resultado en `Docs/dataset_combinado.csv`.

## Qué hace cada etapa

### 1) Loader

- Detecta formato por extensión (`csv`, `json`, `xlsx`).
- Devuelve siempre un `pandas.DataFrame`.
- Lanza excepciones de framework si falta ruta, archivo o formato soportado.

### 2) Validator

- `limpiar_perfil_solicitante(df)`
- `limpiar_historial_transaccional(df)`
- `limpiar_buro_credito(df)`

Cada método devuelve un DataFrame nuevo (no muta el original) con reglas de limpieza ya definidas.

### 3) Preprocessor

- `merge_sources(profile_df, history_df, bureau_df)`

Combina las fuentes por `ID_Cliente`, rellena faltantes y crea la columna `credito_aprobado` en `0`.

## Script de ejemplo

El archivo `ejemplo_pipeline.py` está listo para usarse como base para tus pruebas o para integrarlo con los siguientes módulos del framework.