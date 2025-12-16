# config.py - OPTIMIZADO PARA PRECISIÓN Y BAJO CONSUMO

import os

class Config:
    """Configuración global del proyecto."""
    
    # Modelo a usar: Qwen 2.5 Coder 1.5B (Especializado en lógica y código)
    MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    
    # Configuración de generación 
    MAX_LENGTH = 700  # Más longitud para explicaciones y bloques de código.
    TEMPERATURE = 0.6  # Más bajo para rigor y precisión (ideal para mates/código).
    TOP_P = 0.9  
    TOP_K = 40  
    
    # Sistema
    DEVICE = "cpu"  # Aseguramos el uso de CPU.
    MAX_HISTORY = 5
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    
    @classmethod
    def ensure_dirs(cls):
        """Crear directorios necesarios."""
        os.makedirs(cls.MODELS_DIR, exist_ok=True)