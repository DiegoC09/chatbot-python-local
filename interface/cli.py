from colorama import init, Fore, Style
from chatbot.chatbot import AssistBot
from neuron_demo.neuron import Neuron

init()

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
                user_input = input(Fore.GREEN + "Tú: " + Style.RESET_ALL)
                
                if not user_input.strip():
                    continue
                
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
                
                print(Fore.BLUE + "Bot: " + Style.RESET_ALL, end="", flush=True)
                response = self.bot.chat(user_input)
                print(response)
                print()
                
            except KeyboardInterrupt:
                print(Fore.CYAN + "\n👋 ¡Hasta luego!" + Style.RESET_ALL)
                break
            except Exception as e:
                print(Fore.RED + f"❌ Error: {e}" + Style.RESET_ALL)