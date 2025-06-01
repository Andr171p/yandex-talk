from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent.parent

ENV_PATH = BASE_DIR / ".env"

MODEL_PATH = BASE_DIR / "models" / "vosk-model-small-ru-0.22"
