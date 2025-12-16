# ============================================================================
# ASSISTBOT - Sistema de Chatbot Modular
# ============================================================================
# Proyecto completo listo para usar en Windows 11
# Optimizado para CPU (Ryzen 3200G)
# ============================================================================

# ============================================================================
# ARCHIVO 1: requirements.txt
# ============================================================================
# Copiar este contenido en un archivo llamado "requirements.txt"
"""
torch==2.0.1
transformers==4.35.0
sympy==1.12
colorama==0.4.6
sentencepiece==0.1.99
"""

# ============================================================================
# ARCHIVO 2: config.py
# ============================================================================
# Configuración central del proyecto

import os

class Config:
    """Configuración global del proyecto."""
    
    # Modelo a usar (optimizado para CPU)
    MODEL_NAME = "DeepESP/gpt2-spanish"  # GPT-2 en español
    
    # Configuración de generación
    MAX_LENGTH = 200
    TEMPERATURE = 0.7
    TOP_P = 0.9
    TOP_K = 50
    
    # Sistema
    DEVICE = "cpu"  # Sin GPU, usamos CPU
    MAX_HISTORY = 5  # Últimas 5 interacciones en memoria
    
    # Directorios
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    
    @classmethod
    def ensure_dirs(cls):
        """Crear directorios necesarios."""
        os.makedirs(cls.MODELS_DIR, exist_ok=True)


# ============================================================================
# ARCHIVO 3: neuron_demo/neuron.py
# ============================================================================
# Tu código original mejorado

import math

class Neuron:
    """Neurona artificial con función de activación sigmoid."""
    
    def __init__(self, weights=None, bias=0.0):
        """Inicializar neurona."""
        self.weights = weights
        self.bias = bias
        self.last_output = None
    
    def activate(self, inputs):
        """Activar neurona con entrada."""
        if not inputs:
            raise ValueError("inputs no puede estar vacío")
        
        if self.weights is None:
            self.weights = [1.0] * len(inputs)
        
        if len(inputs) != len(self.weights):
            raise ValueError(
                f"Inputs ({len(inputs)}) != weights ({len(self.weights)})"
            )
        
        # Suma ponderada + bias
        z = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        
        # Sigmoid
        self.last_output = 1 / (1 + math.exp(-z))
        return self.last_output
    
    def __repr__(self):
        return f"Neuron(weights={self.weights}, bias={self.bias})"


class NeuralLayer:
    """Capa de neuronas para demos educativos."""
    
    def __init__(self, n_neurons, n_inputs):
        """Crear capa con n neuronas."""
        self.neurons = [
            Neuron(weights=[1.0] * n_inputs, bias=0.0) 
            for _ in range(n_neurons)
        ]
    
    def forward(self, inputs):
        """Pasar inputs por todas las neuronas."""
        return [neuron.activate(inputs) for neuron in self.neurons]


# ============================================================================
# ARCHIVO 4: chatbot/math_solver.py
# ============================================================================
# Sistema de resolución matemática

import sympy as sp
import re

class MathSolver:
    """Resuelve problemas matemáticos usando SymPy."""
    
    @staticmethod
    def detect_math(text):
        """Detectar si el texto contiene matemática."""
        patterns = [
            r'\d+\s*[\+\-\*/]\s*\d+',  # Operaciones básicas
            r'derivada',
            r'integral',
            r'ecuación',
            r'resolver',
            r'calcular',
        ]
        return any(re.search(p, text.lower()) for p in patterns)
    
    @staticmethod
    def solve(query):
        """Intentar resolver problema matemático."""
        try:
            # Extraer expresión matemática
            expr_match = re.search(r'[\d\+\-\*/\^\(\)x]+', query)
            if not expr_match:
                return None
            
            expr_str = expr_match.group()
            x = sp.Symbol('x')
            
            # Parsear y resolver
            expr = sp.sympify(expr_str)
            
            # Determinar tipo de operación
            if 'derivada' in query.lower():
                result = sp.diff(expr, x)
                return f"La derivada es: {result}"
            elif 'integral' in query.lower():
                result = sp.integrate(expr, x)
                return f"La integral es: {result}"
            else:
                result = sp.simplify(expr)
                return f"Resultado: {result}"
                
        except Exception as e:
            return f"No pude resolver: {str(e)}"


# ============================================================================
# ARCHIVO 5: chatbot/code_helper.py
# ============================================================================
# Asistente de programación

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


# ============================================================================
# ARCHIVO 6: chatbot/model.py
# ============================================================================
# Sistema de modelo de lenguaje

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from config import Config

class ChatModel:
    """Wrapper del modelo de lenguaje."""
    
    def __init__(self):
        """Inicializar modelo."""
        print(f"🤖 Cargando modelo {Config.MODEL_NAME}...")
        print("⏳ Esto puede tardar unos minutos la primera vez...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            Config.MODEL_NAME,
            cache_dir=Config.MODELS_DIR
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_NAME,
            cache_dir=Config.MODELS_DIR,
            torch_dtype=torch.float32  # Para CPU
        )
        
        # Sin pad token, usar eos
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        print("✅ Modelo cargado!\n")
    
    def generate(self, prompt, max_length=None):
        """Generar respuesta."""
        max_length = max_length or Config.MAX_LENGTH
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=max_length,
                temperature=Config.TEMPERATURE,
                top_p=Config.TOP_P,
                top_k=Config.TOP_K,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Limpiar prompt de la respuesta
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        return response


# ============================================================================
# ARCHIVO 7: chatbot/chatbot.py
# ============================================================================
# Sistema principal del chatbot

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
        context = "Eres un asistente útil que ayuda con preguntas generales, matemáticas y programación.\n\n"
        
        # Agregar historial
        for role, msg in self.history[-Config.MAX_HISTORY:]:
            context += f"{role}: {msg}\n"
        
        context += f"Usuario: {user_input}\nAsistente:"
        return context
    
    def chat(self, user_input):
        """Procesar mensaje del usuario."""
        # Agregar a historial
        self.history.append(("Usuario", user_input))
        
        # Detectar tipo de consulta
        if self.math.detect_math(user_input):
            response = self.math.solve(user_input)
            if response:
                self.history.append(("Asistente", response))
                return response
        
        # Consulta de código
        if self.code.detect_code(user_input):
            prompt = self.code.get_code_context() + "\n" + self._build_prompt(user_input)
        else:
            prompt = self._build_prompt(user_input)
        
        # Generar respuesta
        response = self.model.generate(prompt)
        
        # Limpiar respuesta
        response = response.split("Usuario:")[0].strip()
        
        self.history.append(("Asistente", response))
        return response
    
    def reset(self):
        """Limpiar historial."""
        self.history = []


# ============================================================================
# ARCHIVO 8: interface/cli.py
# ============================================================================
# Interfaz de línea de comandos

from colorama import init, Fore, Style
from chatbot.chatbot import AssistBot
from neuron_demo.neuron import Neuron

init()  # Inicializar colorama

class CLI:
    """Interfaz de consola para el chatbot."""
    
    def __init__(self):
        """Inicializar CLI."""
        self.bot = AssistBot()
        self.running = True
    
    def print_header(self):
        """Mostrar banner."""
        print(Fore.CYAN + "=" * 60)
        print("🤖 ASSISTBOT - Tu Asistente Inteligente")
        print("=" * 60 + Style.RESET_ALL)
        print("\nComandos especiales:")
        print("  /demo     - Ver demo de neurona")
        print("  /reset    - Limpiar conversación")
        print("  /salir    - Cerrar chatbot")
        print()
    
    def show_neuron_demo(self):
        """Mostrar demo educativo de neurona."""
        print(Fore.YELLOW + "\n📚 DEMO: Neurona Artificial" + Style.RESET_ALL)
        print("-" * 40)
        
        neuron = Neuron(weights=[0.5, 0.3, 0.2], bias=0.1)
        inputs = [1.0, 2.0, 3.0]
        
        print(f"Inputs:  {inputs}")
        print(f"Weights: {neuron.weights}")
        print(f"Bias:    {neuron.bias}")
        
        output = neuron.activate(inputs)
        print(f"Output:  {output:.4f}")
        print("-" * 40 + "\n")
    
    def run(self):
        """Ejecutar interfaz."""
        self.print_header()
        
        while self.running:
            try:
                # Input del usuario
                user_input = input(Fore.GREEN + "Tú: " + Style.RESET_ALL)
                
                if not user_input.strip():
                    continue
                
                # Comandos especiales
                if user_input.lower() == '/salir':
                    print(Fore.CYAN + "👋 ¡Hasta luego!" + Style.RESET_ALL)
                    break
                
                if user_input.lower() == '/demo':
                    self.show_neuron_demo()
                    continue
                
                if user_input.lower() == '/reset':
                    self.bot.reset()
                    print(Fore.YELLOW + "🔄 Conversación reiniciada" + Style.RESET_ALL)
                    continue
                
                # Procesar con bot
                print(Fore.BLUE + "Bot: " + Style.RESET_ALL, end="", flush=True)
                response = self.bot.chat(user_input)
                print(response)
                print()
                
            except KeyboardInterrupt:
                print(Fore.CYAN + "\n👋 ¡Hasta luego!" + Style.RESET_ALL)
                break
            except Exception as e:
                print(Fore.RED + f"❌ Error: {e}" + Style.RESET_ALL)


# ============================================================================
# ARCHIVO 9: main.py
# ============================================================================
# Punto de entrada principal

from config import Config
from interface.cli import CLI

def main():
    """Función principal."""
    # Crear directorios necesarios
    Config.ensure_dirs()
    
    # Iniciar interfaz
    cli = CLI()
    cli.run()

if __name__ == "__main__":
    main()