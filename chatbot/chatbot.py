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
        # Historial inicial con el Prompt de Sistema
        self.history = [{"role": "system", "content": Config.SYSTEM_PROMPT}]
    
    def chat(self, user_input):
        """Procesar mensaje del usuario."""
        
        # 1. Intentar usar herramienta matemática primero
        if self.math.detect_math(user_input):
            response = self.math.solve(user_input)
            if response:
                self.history.append({"role": "user", "content": user_input})
                self.history.append({"role": "assistant", "content": response})
                return response
        
        # 2. Si no es matemática, preparar el chat normal
        self.history.append({"role": "user", "content": user_input})

        # Usamos el template oficial de Qwen para formatear el diálogo
        # Esto convierte la lista de diccionarios en el texto que el modelo entiende
        prompt = self.model.tokenizer.apply_chat_template(
            self.history,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 3. Generar respuesta
        response = self.model.generate(prompt)
        
        # Limpieza extra: Qwen a veces devuelve todo el texto.
        # Buscamos solo lo nuevo si es necesario, aunque generate() ya intentó limpiar.
        # En Qwen Instruct, lo que importa es que el modelo complete después del header de assistant.
        
        self.history.append({"role": "assistant", "content": response})
        return response
    
    def reset(self):
        """Limpiar historial manteniendo la personalidad."""
        self.history = [{"role": "system", "content": Config.SYSTEM_PROMPT}]