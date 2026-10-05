# Framework ML para Análisis Crediticio

| | |
|---|---|
| **Alumno** | Fernando Daniel Alvarez Jasso |
| **Matrícula** | 2771229 |

Proyecto escolar: un framework en Python para construir un pipeline de
análisis crediticio (carga de datos → validación y limpieza → preprocesamiento →
entrenamiento → evaluación → gráficas para el reporte). El flujo completo está documentado
en [`Docs/diagram.md`](Docs/diagram.md) se usa la sintaxis mermaid.

**Problema que resuelve:** decidir si se aprueba un crédito a un solicitante a partir de tres
fuentes de datos (perfil del solicitante, historial transaccional y buró de crédito). Para eso
se entrenan dos modelos supervisados (Regresión logística y Random forest) y un modelo no
supervisado (K-means) que agrupa a los clientes por perfil.

## Instalación

```bash
pip install -r requirements.txt
```

En **Google Colab**, en una celda del notebook:

```python
!pip install pandas openpyxl matplotlib scikit-learn seaborn
```

(`pandas`, `matplotlib`, `scikit-learn` y `seaborn` suelen venir preinstalados en Colab; `openpyxl` puede
requerir instalación explícita para leer archivos `.xlsx`.)

## Scripts del proyecto: cuál usar y para qué

Todos se ejecutan **desde la raíz del proyecto** (la carpeta donde está este README).

| Script | Para qué sirve | Cómo se ejecuta |
|---|---|---|
| [`main.py`](main.py) | **Flujo completo**: de los archivos de `Docs/` hasta las métricas y las gráficas del reporte. Es el punto de partida recomendado. | `python main.py` |
| [`evaluar_cliente.py`](evaluar_cliente.py) | **Usar el modelo con clientes nuevos**: dice si se aprobaría su crédito, con qué probabilidad y a qué segmento pertenece. | `python evaluar_cliente.py --archivo clientes_nuevos_ejemplo.json` |
| [`ejemplo_modelos.py`](ejemplo_modelos.py) | Ejemplo **detallado de la parte de modelos**: predicciones cliente por cliente, importancia de variables, silhouette y perfiles de segmentos. | `PYTHONPATH=src python ejemplo_modelos.py` |
| [`ejemplo_preprocesador.py`](ejemplo_preprocesador.py) | Ejemplo **de la parte de datos**: limpieza, combinación de las tres fuentes, variable objetivo y preparación para el modelo. No entrena modelos. | `PYTHONPATH=src python ejemplo_preprocesador.py` |

> **En Windows (PowerShell)**, `PYTHONPATH=src` se escribe así:
> ```powershell
> $env:PYTHONPATH = "src"; python ejemplo_modelos.py
> ```
> `main.py` y `evaluar_cliente.py` no lo necesitan: agregan `src/` al path por sí solos.

### 1. `main.py`: flujo completo

```bash
python main.py
```

Ejecuta todo el pipeline en orden y lo imprime en consola en 9 pasos numerados. Cada paso del
código tiene un comentario que explica qué componente se llama y para qué:

| Paso | Componente | Qué hace |
|---|---|---|
| 1 | `DataLoader` | Carga `perfil_solicitante.csv`, `historial_transaccional.json` y `buro_credito.xlsx` |
| 2 | `DataValidator.revisar_calidad` | Revisa estructura, faltantes, duplicados y edades atípicas |
| 3 | `DataValidator.limpiar_*` | Limpia cada fuente (duplicados, "ND", USD → MXN, morosidad normalizada) |
| 4 | `Preprocessor.merge_sources` | Combina las tres fuentes en una tabla por `ID_Cliente` |
| 5 | `Preprocessor.define_credit_approval` | Calcula la variable objetivo `credito_aprobado` con la regla de negocio |
| 6 | `Preprocessor.prepare_for_model` | Divide 80/20, imputa, codifica y escala |
| 7 | `ModelTrainer` | Entrena Regresión logística, Random forest y K-means |
| 8 | `ModelEvaluator` | Métricas de prueba, validación cruzada, recomendación y segmentación |
| 9 | `ReportFigures` | Guarda las gráficas del reporte en `reporte/figuras/` |

Resultado del paso 8 con los datos de `Docs/` (fragmento de la salida real):

```text
                  Regresión logística  Random forest
precision_prueba                0.571          0.818
recall_prueba                   0.444          1.000
f1_prueba                       0.500          0.900
auc_prueba                      0.833          0.910
precision_cv                    0.587          0.837
f1_cv                           0.493          0.806

Modelo recomendado: Random forest

Segmentación: silhouette 0.15, k sugerido = 3
```

### 2. `evaluar_cliente.py`: evaluar clientes nuevos

Simula el uso real del framework: llega un solicitante nuevo y se quiere saber si se le
aprobaría el crédito. El script:

1. Entrena los modelos con los datos de `Docs/` (igual que `main.py`).
2. Aplica al cliente nuevo **la misma limpieza, imputación y escalado** aprendidos en
   entrenamiento. Si le faltan datos, los rellena con los valores de entrenamiento y avisa
   cuáles.
3. Muestra para cada cliente:
   - El resultado de la **regla de negocio** y, si no la cumple, **qué condiciones fallan**.
   - La **predicción y probabilidad de aprobación** de cada modelo, marcando el recomendado.
   - El **segmento K-means** al que pertenece y el perfil típico de ese segmento.

**Opción A: un cliente con argumentos** (una sola transacción):

```bash
python evaluar_cliente.py --edad 35 --ingreso 25000 --antiguedad 4.5 --vivienda Propia \
    --monto 1200 --saldo 15400 --frecuencia 4 --score 720 --creditos 2 --morosidad NO
```

En PowerShell, escribe el comando en una sola línea o usa `` ` `` en lugar de `\` al final de cada línea.

| Argumento | Significado | Fuente |
|---|---|---|
| `--edad` | Edad en años | Perfil |
| `--ingreso` | Ingreso declarado mensual | Perfil |
| `--antiguedad` | Antigüedad laboral en años | Perfil |
| `--vivienda` | `Propia`, `Rentada`, `Hipotecada`, ... | Perfil |
| `--monto` | Monto de la transacción | Historial |
| `--moneda` | `MXN` o `USD` (default `MXN`) | Historial |
| `--saldo` | Saldo promedio mensual | Historial |
| `--frecuencia` | Frecuencia de depósito | Historial |
| `--score` | Score de buró | Buró |
| `--creditos` | Créditos activos | Buró |
| `--morosidad` | `SI` / `NO` (default `NO`) | Buró |
| `--salida` | (Opcional) CSV donde guardar el resumen | — |

Salida (fragmento):

```text
Regla de negocio: APROBADO

Modelos:
  Regresión logística    APROBADO     probabilidad de aprobación 0.60
  Random forest          APROBADO     probabilidad de aprobación 0.74  <- recomendado

Segmento K-means: 0
```

**Opción B: varios clientes desde un JSON**:

```bash
python evaluar_cliente.py --archivo clientes_nuevos_ejemplo.json --salida resultados.csv
```

[`clientes_nuevos_ejemplo.json`](clientes_nuevos_ejemplo.json) trae tres casos de prueba: un
cliente que cumple todo, uno con score bajo, morosidad y montos en USD, y uno con datos
incompletos y sin transacciones. Cada cliente tiene las claves `perfil`, `historial` (lista de
transacciones) y `buro`, con las mismas columnas que los archivos de `Docs/`:

```json
{
    "ID_Cliente": "NUEVO-001",
    "perfil": {"Edad": 35, "Ingreso_Declarado": 32000, "Antiguedad_Laboral": 6, "Tipo_Vivienda": "Propia"},
    "historial": [
        {"Monto_Transaccion": 1800, "Moneda": "MXN", "Saldo_Promedio_Mensual": 16000, "Frecuencia_Deposito": 5}
    ],
    "buro": {"Score_Buro": 720, "Creditos_Activos": 1, "Mantiene_Morosidad_Previa": "NO"}
}
```

Resumen que imprime al final:

```text
ID_Cliente  regla_negocio  Regresión logística (prob)  Random forest (pred)  Random forest (prob)  segmento
 NUEVO-001              1                        0.71                     1                  0.70         2
 NUEVO-002              0                        0.02                     0                  0.02         1
 NUEVO-003              0                        0.16                     0                  0.42         0
```

### 3. `ejemplo_modelos.py`: detalle de entrenamiento y evaluación

```bash
PYTHONPATH=src python ejemplo_modelos.py
```

Recorre el mismo flujo que `main.py`, pero con menos explicación de los datos y **más
detalle de los modelos**. Sirve para revisar y documentar los resultados en el reporte:

- Tabla de modelos entrenados con hiperparámetros y tiempo de entrenamiento.
- **Predicción y probabilidad de cada cliente del conjunto de prueba** con ambos modelos,
  junto al valor real.
- Comparación completa de métricas (prueba y validación cruzada) y la justificación del
  modelo recomendado.
- **Importancia de variables** de cada modelo.
- Silhouette para k = 2 a 5 y **perfil de cada segmento** en escala original (edad en años,
  ingreso en pesos).
- Genera las gráficas en `reporte/figuras/` e imprime su interpretación escrita.

### 4. `ejemplo_preprocesador.py`: detalle de la preparación de datos

```bash
PYTHONPATH=src python ejemplo_preprocesador.py
```

Muestra **solo la parte de datos**, sin entrenar modelos. Sirve para ver cómo quedan los
datos antes de llegar al modelo:

- Carga y limpia las tres fuentes.
- Imprime la **tabla combinada** por cliente con la variable `credito_aprobado`.
- Cuenta los clientes por clase (aprobados / no aprobados).
- Guarda la tabla combinada como imagen en [`tabla_combinada.png`](tabla_combinada.png).
- Prepara los conjuntos de entrenamiento y prueba e imprime sus dimensiones, las variables
  que usa el modelo y las que se descartaron.

## Estructura del proyecto

```text
python-framework-crediticio/
├── main.py                       # Flujo completo (punto de partida)
├── evaluar_cliente.py            # Evaluar clientes nuevos con los modelos
├── ejemplo_modelos.py            # Ejemplo detallado de modelos y evaluación
├── ejemplo_preprocesador.py      # Ejemplo detallado de preparación de datos
├── clientes_nuevos_ejemplo.json  # Clientes de prueba para evaluar_cliente.py
├── Docs/                         # Datos de entrada y documentación técnica
├── src/
│   ├── data_loader/              # Carga CSV, JSON y XLSX
│   ├── data_validator/           # Calidad y limpieza de datos
│   ├── preprocessor/             # Fusión, variable objetivo, imputación, escalado
│   ├── model_trainer/            # Regresión logística, Random forest, K-means
│   ├── model_evaluator/          # Métricas, validación cruzada, silhouette
│   ├── data_visualizer/          # Gráficas del reporte
│   └── exceptions.py             # Excepciones del framework
├── reporte/figuras/              # Gráficas generadas
├── scripts/                      # Generador de clientes sintéticos
├── specs/                        # Especificación y diseño de cada componente
└── tests/                        # Pruebas automatizadas (pytest)
```

## Componentes disponibles

### `DataLoader`

Carga archivos de datos en formato CSV, JSON o XLSX a través de un único
punto de entrada. No tiene interfaz visual ni de consola: es una API
puramente programática, pensada para usarse desde código (incluyendo
notebooks de Google Colab).

```python
from data_loader import DataLoader

loader = DataLoader()
df = loader.load("Docs/perfil_solicitante.csv")   # también funciona con .json y .xlsx
```

Documentación completa del componente: [`specs/001-dataloader-multiformat/`](specs/001-dataloader-multiformat/)
(`spec.md`, `plan.md`, `contracts/dataloader-api.md`, `quickstart.md`).

### `DataValidator`

Limpia y normaliza los `DataFrame` que produce `DataLoader` para las tres
fuentes de datos del proyecto (buró de crédito, historial transaccional,
perfil del solicitante). Igual que `DataLoader`, es una API puramente
programática: recibe y devuelve `pandas.DataFrame`.

```python
from data_loader import DataLoader
from data_validator import DataValidator

loader = DataLoader()
validator = DataValidator()  # tasa_cambio_usd_mxn=15 por defecto

df_buro = loader.load("Docs/buro_credito.xlsx")
df_buro_limpio = validator.limpiar_buro_credito(df_buro)

df_historial = loader.load("Docs/historial_transaccional.json")
df_historial_limpio = validator.limpiar_historial_transaccional(df_historial)

df_perfil = loader.load("Docs/perfil_solicitante.csv")
df_perfil_limpio = validator.limpiar_perfil_solicitante(df_perfil)
```

Además de normalizar, cada método de limpieza:

- **Valida la estructura**: si falta una columna esperada de la fuente, lanza
  `MissingColumnsError` con el nombre de la fuente y de las columnas faltantes.
- **Elimina duplicados**: quita filas idénticas (después de normalizar) y, en perfil y
  buró, deja un solo registro por `ID_Cliente` (el primero).
- **Marca faltantes como "ND"**: los ingresos, antigüedades, montos y saldos vacíos
  quedan como `NaN` (no como `0`), y `DataVisualizer.mostrar_tabla` los muestra como `ND`.
- **Trata la edad atípica**: una `Edad` menor a 18, mayor a 70, vacía o no numérica
  queda como `NaN` ("ND") sin eliminar el registro.

Para revisar la estructura y calidad de una fuente antes y después de limpiarla:

```python
antes = validator.revisar_calidad(df_perfil, "perfil_solicitante")
despues = validator.revisar_calidad(df_perfil_limpio, "perfil_solicitante")

print(antes.edades_atipicas, despues.edades_atipicas)  # 9 0
print(despues.como_tabla())                            # tipo y faltantes por columna
```

`revisar_calidad` devuelve un `ReporteCalidad` con filas, columnas faltantes y
adicionales, tipos, faltantes por columna, duplicados, `ID_Cliente` repetidos y edades
atípicas. Nunca modifica los datos ni lanza error por datos sucios.

Para leer el reporte sin conocer sus atributos:

```python
print(antes.resumen())                                   # texto legible de una fuente
print(DataValidator.comparar_calidad(antes, despues))    # tabla antes/después por indicador
```

```text
Reporte de calidad: perfil_solicitante
Filas: 138 | Estructura válida: sí
Duplicados: 3 | IDs repetidos: 3 | Edades atípicas: 9
Faltantes: Ingreso_Declarado: 5, Antiguedad_Laboral: 7
```

Los "ND" se conservan también al combinar las fuentes en `Preprocessor.merge_sources`.

Documentación completa del componente: [`specs/002-datavalidator-limpieza/`](specs/002-datavalidator-limpieza/)
y [`specs/006-datavalidator-calidad-datos/`](specs/006-datavalidator-calidad-datos/)
(`spec.md`, `plan.md`, `contracts/datavalidator-api.md`, `quickstart.md`).

### `DataVisualizer`

Genera visualizaciones exploratorias (con `matplotlib`) a partir de los `DataFrame` ya
limpiados por `DataValidator`: una tabla de cualquiera de las tres fuentes, la relación
entre créditos activos y score de buró, la relación entre score de buró e ingreso
declarado (cruzando buró y perfil por `ID_Cliente`), y la distribución de tipo de
vivienda. Cada método devuelve un `matplotlib.figure.Figure` en vez de mostrarlo
directamente.

```python
from data_visualizer import DataVisualizer

visualizer = DataVisualizer()

fig_tabla = visualizer.mostrar_tabla(df_buro_limpio, titulo="Buró de crédito (limpio)")
fig_creditos_score = visualizer.graficar_creditos_vs_score(df_buro_limpio)
fig_score_ingreso = visualizer.graficar_score_vs_ingreso(df_buro_limpio, df_perfil_limpio)
fig_vivienda = visualizer.graficar_distribucion_tipo_vivienda(df_perfil_limpio)
# En un notebook, devolver la Figure como última expresión de una celda la muestra.
```

Documentación completa del componente: [`specs/003-datavisualizer-graficas/`](specs/003-datavisualizer-graficas/)
(`spec.md`, `plan.md`, `contracts/datavisualizer-api.md`, `quickstart.md`).

#### Gráficas analíticas (feature 011)

Gráficas con Seaborn para el reporte, cada una con título, etiquetas, leyenda y una
**interpretación escrita generada a partir de los datos**:

| Identificador | Qué muestra |
|---|---|
| `calidad_<fuente>` (×3) | Faltantes por columna, duplicados y edades atípicas, antes y después de la limpieza |
| `distribucion_clases` | Clientes aprobados y no aprobados (balance de clases) |
| `correlaciones` | Mapa de calor de correlaciones entre las variables y `credito_aprobado` |
| `matriz_confusion` | Aciertos y tipos de error de cada modelo en el conjunto de prueba |
| `curva_roc` | Comparación de la capacidad de separar aprobados de no aprobados (AUC) |
| `importancia_variables` | Qué variables pesan más en cada modelo; resalta las de la regla |
| `silhouette` | Calidad de la segmentación para 2 a 5 segmentos |
| `pca_segmentos` | Mapa 2D de los clientes coloreado por segmento |

```python
from data_visualizer import ReportFigures

graficas = ReportFigures.generar(calidad, preprocessor.aggregate_by_client(df_etiquetado),
                                 resultado, evaluacion)
for g in ReportFigures.guardar_todas(graficas, "reporte/figuras"):
    print(g.ruta, g.interpretacion)
```

(`calidad` es un `dict` fuente → `(revisar_calidad(original), revisar_calidad(limpio))`; ver
el ejemplo completo en `ejemplo_modelos.py`.)

- **Colores**: tomados de una paleta validada para daltonismo. La regresión logística es
  siempre azul y el random forest siempre naranja, en todas las gráficas.
- **Interpretaciones**: describen lo que muestran los datos (valores, máximos,
  comparaciones). Son una **base** para el reporte; el análisis y las conclusiones de
  negocio los redacta el estudiante.

Documentación: [`specs/011-datavisualizer-seaborn/`](specs/011-datavisualizer-seaborn/).

### `Preprocessor`

Combina las tres fuentes ya limpias en una sola tabla por `ID_Cliente` (`merge_sources`).
Entran todos los clientes de cualquier fuente, y lo que a un cliente le falta de otra fuente
queda como "ND" (`NaN`), sin rellenarse con `0`. Además, calcula la variable objetivo `credito_aprobado` con la regla de negocio del proyecto
(`define_credit_approval`). Un cliente queda aprobado (`1`) solo si cumple **las cuatro**
condiciones:

1. Tiene entre **18 y 69 años** (ambos incluidos).
2. Su `Score_Buro` es **mayor que 600**.
3. **No** tiene morosidad previa (`Mantiene_Morosidad_Previa == 0`).
4. **Ha recibido transacciones**: al menos una fila con `Frecuencia_Deposito > 0`.

En cualquier otro caso queda en `0`. Un dato faltante nunca cuenta como condición cumplida.
Los umbrales son constantes de la clase (`EDAD_MINIMA_APROBACION`,
`EDAD_MAXIMA_APROBACION`, `SCORE_MINIMO_APROBACION`).

```python
from preprocessor import Preprocessor

preprocessor = Preprocessor()
df_combinado = preprocessor.merge_sources(df_perfil_limpio, df_historial_limpio, df_buro_limpio)
df_etiquetado = preprocessor.define_credit_approval(df_combinado)

print(preprocessor.approval_summary(df_etiquetado))  # clientes por clase: 0 → 88, 1 → 47
```

Ejemplo completo de punta a punta: `PYTHONPATH=src python ejemplo_preprocesador.py`.

#### Preparación para el modelo

`prepare_for_model` convierte la tabla etiquetada en los conjuntos que recibe `ModelTrainer`.
Cada paso es una clase en su propio archivo dentro de `src/preprocessor/`:

| Paso | Clase | Archivo | Qué hace |
|---|---|---|---|
| 1 | `Preprocessor.aggregate_by_client` | `preprocessor.py` | Una fila por cliente: monto total, saldo promedio, frecuencia máxima y número de transacciones |
| 2 | `TrainTestSplitter` | `splitter.py` | 80% entrenamiento / 20% prueba, conservando la proporción de aprobados (semilla 42) |
| 3 | `DataImputer` | `imputer.py` | Rellena los "ND" con la mediana (o la moda en texto) **de entrenamiento** |
| 4 | `CategoricalEncoder` | `encoder.py` | Convierte columnas de texto en columnas 0/1 |
| 5 | `FeatureScaler` | `scaler.py` | Escala a media 0 y desviación 1 (las variables 0/1 no se escalan) |

Los pasos 3 a 5 se ajustan (`fit`) **solo** con el conjunto de entrenamiento y se aplican
(`transform`) igual a entrenamiento y prueba, para no filtrar información del conjunto de
prueba al modelo.

```python
datos = preprocessor.prepare_for_model(df_etiquetado)
X_train, X_test, y_train, y_test = datos.as_tuple()   # (108, 11) y (27, 11) con Docs/

print(datos.variables)           # variables que usa el modelo
print(datos.descartadas)         # {'Moneda': 'moneda única (MXN)'}
print(datos.valores_imputacion)  # valor con el que se rellenó cada "ND"
```

> Aquí **no** se entrena ningún modelo: eso le corresponde a `ModelTrainer`.
> Las variables con las que se definió `credito_aprobado` (edad, score, morosidad y
> frecuencia de depósito) sí entran al modelo, por decisión del proyecto. Las métricas
> medirán qué tan bien el modelo recupera esa regla de negocio.

Documentación completa del componente: [`specs/005-preprocesador-fusion-clientes/`](specs/005-preprocesador-fusion-clientes/),
[`specs/007-preprocessor-credito-aprobado/`](specs/007-preprocessor-credito-aprobado/) y
[`specs/008-preprocesamiento-modelo/`](specs/008-preprocesamiento-modelo/).

### `ModelTrainer`

Entrena los modelos del proyecto a partir de los datos que prepara
`Preprocessor.prepare_for_model`:

| Modelo | Tipo | Por qué se eligió |
|---|---|---|
| Regresión logística | Supervisado | Estándar en scoring de crédito; lineal e interpretable (sus coeficientes indican cuánto pesa cada variable) |
| Random forest | Supervisado | Combina árboles de decisión, así que aprende reglas con varios umbrales, como la regla de aprobación; indica la importancia de cada variable |
| K-means (3 segmentos) | No supervisado | Agrupa a los clientes por perfil **sin ver la etiqueta**; puede revelar grupos que la regla no distingue |

Cada modelo es una clase en su propio archivo dentro de `src/model_trainer/`:

| Archivo | Clase | Rol |
|---|---|---|
| `base.py` | `SupervisedModel` | Base común: valida datos, mide el tiempo, `fit` / `predict` / `predict_proba` |
| `logistic_model.py` | `LogisticRegressionModel` | Regresión logística (hereda de `SupervisedModel`) |
| `random_forest_model.py` | `RandomForestModel` | Random forest, 100 árboles (hereda de `SupervisedModel`) |
| `segmentation.py` | `ClientSegmentation` | K-means |
| `training_result.py` | `TrainingResult` | Resultado: modelos, segmento de cada cliente y `resumen()` |
| `trainer.py` | `ModelTrainer` | Entrena los tres modelos en un solo paso |

```python
from model_trainer import ModelTrainer

resultado = ModelTrainer().train(datos)        # datos = preprocessor.prepare_for_model(...)
print(resultado.resumen())                     # modelos, hiperparámetros y tiempo de entrenamiento
modelo = resultado.modelos["Random forest"]
print(modelo.predict(datos.X_test))            # 0/1 por cliente de prueba
print(modelo.predict_proba(datos.X_test))      # probabilidad de aprobación
print(resultado.segmentos.value_counts())      # clientes por segmento (índice: ID_Cliente)
```

Todo es reproducible (semilla 42). El `ModelTrainer` **no** calcula métricas ni compara
modelos: eso le corresponde a `ModelEvaluator`.

Ejemplo completo de punta a punta: `PYTHONPATH=src python ejemplo_modelos.py`.

Documentación completa del componente: [`specs/009-model-trainer/`](specs/009-model-trainer/).

### `ModelEvaluator`

Evalúa los modelos que entrena `ModelTrainer` y la segmentación de clientes:

| Métrica | Qué mide y por qué importa en crédito |
|---|---|
| Exactitud | Proporción de clientes bien clasificados; trata igual los dos tipos de error |
| **Precisión** | De los que el modelo aprueba, cuántos cumplen la política. **Métrica principal**: aprobar a quien no cumple (falso positivo) es el error más costoso |
| Recall | De los que cumplen, cuántos aprueba el modelo; un recall bajo es negocio perdido |
| F1 | Equilibrio entre precisión y recall; se usa como desempate |
| AUC-ROC | Qué tan bien ordena a los clientes por probabilidad, sin depender del umbral 0.5 |
| Matriz de confusión | Cuántos aciertos y de qué tipo son los errores |
| Silhouette | Qué tan separados están los segmentos de K-means (de -1 a 1) |

(El texto completo está en `DESCRIPCION_METRICAS`, para usarlo en el reporte.)

**Criterio de recomendación**: mayor precisión media en validación cruzada (5 particiones);
si empatan, mayor F1; si siguen empatados, la regresión logística (más interpretable). La
recomendación avisa si la diferencia entre modelos **no es significativa** (menor que la
variación entre particiones) y si el modelo elegido rechaza a más de la mitad de los buenos
clientes (recall < 0.5).

| Archivo (`src/model_evaluator/`) | Clase | Qué hace |
|---|---|---|
| `metrics.py` | `ClassificationMetrics` | Métricas de prueba y sus descripciones |
| `cross_validation.py` | `CrossValidator` | Validación cruzada estratificada (con copias de los modelos) |
| `comparison.py` | `ModelComparator` | Tabla comparativa y `Recomendacion` |
| `importance.py` | `FeatureImportance` | Coeficientes / importancias, marcando las variables de la regla |
| `segmentation_evaluator.py` | `SegmentationEvaluator` | Silhouette por número de segmentos y perfiles en escala original |
| `evaluator.py` | `ModelEvaluator` | Evalúa todo en un paso → `EvaluationResult` |

```python
from model_evaluator import ModelEvaluator

evaluacion = ModelEvaluator().evaluate(resultado)   # resultado = ModelTrainer().train(datos)
print(evaluacion.comparacion.round(3))
print(evaluacion.recomendacion.justificacion)
print(evaluacion.importancias["Regresión logística"])
print(evaluacion.silhouette, evaluacion.k_sugerido)
print(evaluacion.perfiles_segmentos.round(2))
```

Ejemplo completo: `PYTHONPATH=src python ejemplo_modelos.py`. Documentación:
[`specs/010-model-evaluator/`](specs/010-model-evaluator/) (incluye los hallazgos con los
datos de `Docs/` en `research.md`, R8).

### Excepciones

Las excepciones del framework están centralizadas en un único módulo,
[`src/exceptions.py`](src/exceptions.py), con una clase base común
(`FrameworkError`) de la que heredan las excepciones específicas de cada
componente (por ejemplo, `DataLoaderError`, `DataValidatorError` y `PreprocessorError`,
con sus subclases como `ModelPreparationError`, `ModelTrainerError` y `ModelEvaluatorError`).

## Datos de ejemplo

`Docs/` contiene tres fuentes con **135 clientes**:
