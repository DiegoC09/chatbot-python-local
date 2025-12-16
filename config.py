import os

class Config:
    """Configuración global del proyecto."""
    
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Configuración de generación (AJUSTADA)
    MAX_LENGTH = 100  # Acortamos
    TEMPERATURE = 0.8  # Más creativo
    TOP_P = 0.92  # Más variado
    TOP_K = 40  # Menos repetitivo
    NO_REPEAT_NGRAM_SIZE = 3  # ← NUEVA LÍNEA
    
    DEVICE = "cpu"
    MAX_HISTORY = 5
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    
    @classmethod
    def ensure_dirs(cls):
        """Crear directorios necesarios."""
        os.makedirs(cls.MODELS_DIR, exist_ok=True)