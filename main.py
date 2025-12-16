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