import os

class Config:
    """Configuración global del proyecto."""
    
    # Modelo: Qwen 2.5 Coder 1.5B (Ideal para tu RAM y programar)
    MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    
    # Configuración de generación 
    MAX_LENGTH = 700  # Más espacio para explicar código
    TEMPERATURE = 0.6  # Precisión técnica
    TOP_P = 0.9
    TOP_K = 40
    
    # Personalidad del Bot
    SYSTEM_PROMPT = (
        "Eres un asistente experto en programación, lógica y matemáticas. "
        "Tu tono es directo, técnico y en español. Si te piden código, coméntalo bien."
    )
    
    # Sistema
    DEVICE = "cpu"
    MAX_HISTORY = 10  # Aumentamos un poco el historial
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    
    @classmethod
    def ensure_dirs(cls):
        """Crear directorios necesarios."""
        os.makedirs(cls.MODELS_DIR, exist_ok=True)