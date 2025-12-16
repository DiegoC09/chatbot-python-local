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