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
            torch_dtype=torch.float32
        )
        
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