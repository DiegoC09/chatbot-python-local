from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from config import Config

class ChatModel:
    """Wrapper del modelo de lenguaje."""
    
    def __init__(self):
        """Inicializar modelo con optimización de memoria."""
        print(f"🤖 Cargando modelo {Config.MODEL_NAME}...")
        print("💡 Optimizando para 8GB RAM (Bfloat16)...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            Config.MODEL_NAME,
            cache_dir=Config.MODELS_DIR
        )
        
        # Corrección de token de padding
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Carga optimizada para CPU Ryzen + Poca RAM
        self.model = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_NAME,
            cache_dir=Config.MODELS_DIR,
            dtype=torch.bfloat16,        # Usa la mitad de memoria que float32
            low_cpu_mem_usage=True,      # Carga inteligente (requiere accelerate)
        )
        
        self.model.eval()
        print("✅ Modelo cargado y optimizado!\n")
    
    def generate(self, prompt, max_length=None):
        """Generar respuesta."""
        max_length = max_length or Config.MAX_LENGTH
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048 # Contexto seguro
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                temperature=Config.TEMPERATURE,
                top_p=Config.TOP_P,
                top_k=Config.TOP_K,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Limpieza básica: quitar el prompt si el modelo lo repite
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        # Limpieza avanzada para Qwen (a veces repite el system prompt)
        if "system" in response and "user" in response:
             parts = response.split("assistant")
             if len(parts) > 1:
                 return parts[-1].strip()

        return response