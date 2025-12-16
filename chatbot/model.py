# chatbot/model.py - OPTIMIZADO PARA MEMORIA (BFLOAT16 Y ACELERACIÓN)

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from config import Config
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

class ChatModel:
    """Wrapper del modelo de lenguaje."""
    
    def __init__(self):
        """Inicializar modelo con optimización de memoria."""
        print(f"🤖 Cargando modelo {Config.MODEL_NAME}...")
        print("💡 Optimizando para 8GB RAM (Bfloat16)...")
        
        # 1. Cargar Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            Config.MODEL_NAME,
            cache_dir=Config.MODELS_DIR
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2. Cargar el modelo con la optimización de precisión
        self.model = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_NAME,
            cache_dir=Config.MODELS_DIR,
            torch_dtype=torch.bfloat16,  # CLAVE: Reduce RAM a la mitad (mejor para CPU)
            low_cpu_mem_usage=True,      # CLAVE: Usa accelerate para cargar de forma eficiente
        )
        
        self.model.eval()
        print("✅ Modelo cargado y optimizado!\n")
    
    def generate(self, prompt, max_length=None):
        """Generar respuesta."""
        max_length = max_length or Config.MAX_LENGTH
        
        # El resto de la función es similar... (No te la copio toda, solo la parte del __init__ es la crítica)
        # Asegurate de que la función generate en tu archivo original sigue bien:
        # 1. inputs = self.tokenizer(...)
        # 2. with torch.no_grad(): outputs = self.model.generate(...)
        # 3. response = self.tokenizer.decode(...)
        #
        # Nota: Vi que en tu archivo subido agregaste 'attention_mask', 'no_repeat_ngram_size' y 'repetition_penalty'. ¡Excelente! Dejalos.
        
        # ... (Cuerpo de generate) ...
        # (Usaré el cuerpo que tienes en tu archivo 'chatbot/model.py' para completar el resto de la función, ya que está bien)
        
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
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                temperature=Config.TEMPERATURE,
                top_p=Config.TOP_P,
                top_k=Config.TOP_K,
                do_sample=True,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        return response