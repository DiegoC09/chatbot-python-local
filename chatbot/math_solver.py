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