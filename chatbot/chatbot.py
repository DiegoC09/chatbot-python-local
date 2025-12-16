from chatbot.model import ChatModel
from chatbot.math_solver import MathSolver
from chatbot.code_helper import CodeHelper
from config import Config

class AssistBot:
    """Chatbot principal con capacidades multimodales."""
    
    def __init__(self):
        """Inicializar chatbot."""
        self.model = ChatModel()
        self.math = MathSolver()
        self.code = CodeHelper()
        self.history = []
    
def _build_prompt(self, user_input):
    """Construir prompt con contexto."""
    # Formato especial para TinyLlama
    context = "<|system|>\nEres un asistente útil que ayuda con preguntas generales, matemáticas y programación.</s>\n"
    
    # Agregar historial
    for role, msg in self.history[-Config.MAX_HISTORY:]:
        if role == "Usuario":
            context += f"<|user|>\n{msg}</s>\n"
        else:
            context += f"<|assistant|>\n{msg}</s>\n"
    
    context += f"<|user|>\n{user_input}</s>\n<|assistant|>\n"
    return context
    
    def chat(self, user_input):
        """Procesar mensaje del usuario."""
        self.history.append(("Usuario", user_input))
        
        if self.math.detect_math(user_input):
            response = self.math.solve(user_input)
            if response:
                self.history.append(("Asistente", response))
                return response
        
        if self.code.detect_code(user_input):
            prompt = self.code.get_code_context() + "\n" + self._build_prompt(user_input)
        else:
            prompt = self._build_prompt(user_input)
        
        response = self.model.generate(prompt)
        response = response.split("Usuario:")[0].strip()
        
        self.history.append(("Asistente", response))
        return response
    
    def reset(self):
        """Limpiar historial."""
        self.history = []