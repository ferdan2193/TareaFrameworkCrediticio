sequenceDiagram
    autonumber

    actor Usuario

    participant DL as DataLoader
    participant DV as DataValidator
    participant PP as Preprocessor
    participant MT as ModelTrainer
    participant ME as ModelEvaluator
    participant VIS as DataVisualizer

    Usuario->>DL: Proporciona archivos CSV, JSON y XLSX<br/>(perfil, historial, buró)

    activate DL
    DL->>DL: Detecta el formato por la extensión
    DL->>DL: Lee el archivo con el lector correspondiente
    DL-->>DV: DataFrame crudo (por fuente)
    deactivate DL

    activate DV
    DV->>DV: Revisa estructura y calidad (antes)
    DV->>DV: Valida columnas esperadas
    DV->>DV: Normaliza formatos (morosidad 0/1, USD→MXN,<br/>prefijo de ID, tipo de vivienda)
    DV->>DV: Marca faltantes y edades atípicas como "ND"
    DV->>DV: Elimina duplicados
    DV->>DV: Revisa estructura y calidad (después)
    DV-->>PP: DataFrame limpio (por fuente)
    DV-->>VIS: ReporteCalidad antes / después
    deactivate DV

    activate PP
    PP->>PP: Combina las fuentes por ID_Cliente (unión, sin relleno)
    PP->>PP: Define credito_aprobado con la regla de negocio
    PP->>PP: Agrega a una fila por cliente
    PP->>PP: Separa entrenamiento y prueba (estratificado)
    PP->>PP: Imputa "ND" (mediana de entrenamiento)
    PP->>PP: Codifica variables de texto
    PP->>PP: Escala variables numéricas
    PP-->>MT: ModelData (X_train, X_test, y_train, y_test)
    PP-->>VIS: Tabla por cliente (distribución y correlaciones)
    deactivate PP

    activate MT
    MT->>MT: Entrena Regresión logística (supervisado)
    MT->>MT: Entrena Random forest (supervisado)
    MT->>MT: Entrena K-means con X_train, sin la etiqueta (no supervisado)
    MT->>MT: Asigna un segmento a cada cliente
    MT-->>ME: TrainingResult (modelos entrenados + segmentos)
    deactivate MT

    activate ME
    ME->>ME: Predice utilizando X_test
    ME->>ME: Compara predicción con y_test y calcula métricas
    ME->>ME: Validación cruzada (5 particiones)
    ME->>ME: Compara modelos y recomienda uno (precisión → F1)
    ME->>ME: Importancia de variables
    ME->>ME: Silhouette y perfiles de segmentos
    ME-->>VIS: EvaluationResult
    deactivate ME

    activate VIS
    VIS->>VIS: Genera 11 gráficas con interpretación escrita<br/>(calidad, exploratorias, modelos, segmentación)
    VIS-->>Usuario: Gráficas PNG en reporte/figuras/
    deactivate VIS

    ME-->>Usuario: Métricas, comparación y modelo recomendado

    Note over Usuario: El reporte final lo redacta el estudiante<br/>a partir de las métricas y las gráficas.<br/>Ejecución completa: python main.py
