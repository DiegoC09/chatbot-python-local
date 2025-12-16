import re

class CodeHelper:
    """Ayuda con código y programación."""
    
    KEYWORDS = ['código', 'programa', 'función', 'class', 'def', 'python']
    
    @staticmethod
    def detect_code(text):
        """Detectar si pregunta sobre código."""
        return any(kw in text.lower() for kw in CodeHelper.KEYWORDS)
    
    @staticmethod
    def get_code_context():
        """Retornar contexto para programación."""
        return (
            "Soy un asistente de programación. "
            "Puedo ayudarte con Python, estructuras de datos, "
            "algoritmos y debugging."
        )