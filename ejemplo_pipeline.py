from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"

# Permite importar los paquetes del framework desde la carpeta src.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_loader import DataLoader
from data_validator import DataValidator
from preprocessor import Preprocessor
from exceptions import FrameworkError


def main() -> None:
    docs_dir = ROOT_DIR / "Docs"

    profile_path = docs_dir / "perfil_solicitante.csv"
    history_path = docs_dir / "historial_transaccional.json"
    bureau_path = docs_dir / "buro_credito.xlsx"

    print("Iniciando pipeline...")

    loader = DataLoader()
    validator = DataValidator(tasa_cambio_usd_mxn=15)
    preprocessor = Preprocessor()

    # 1) Carga
    profile_raw = loader.load(str(profile_path))
    history_raw = loader.load(str(history_path))
    bureau_raw = loader.load(str(bureau_path))

    # 2) Validación/Limpieza
    profile_clean = validator.limpiar_perfil_solicitante(profile_raw)
    history_clean = validator.limpiar_historial_transaccional(history_raw)
    bureau_clean = validator.limpiar_buro_credito(bureau_raw)

    # 3) Preprocesamiento/Fusión
    combined_df = preprocessor.merge_sources(profile_clean, history_clean, bureau_clean)

    output_path = docs_dir / "dataset_combinado.csv"
    combined_df.to_csv(output_path, index=False)

    print(f"Pipeline finalizado. Filas: {len(combined_df)}, columnas: {len(combined_df.columns)}")
    print(f"Archivo generado: {output_path}")
    print("Vista previa:")
    print(combined_df.head(5).to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except FrameworkError as exc:
        print(f"Error del framework: {exc}")
        raise
    except KeyError as exc:
        print(f"Falta una columna esperada en los archivos de entrada: {exc}")
        raise