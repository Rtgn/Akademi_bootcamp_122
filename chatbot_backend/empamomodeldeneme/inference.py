import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import os
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class DietExerciseModel:
    def __init__(self):
        """Model sınıfını başlatır ama henüz yüklemez"""
        self.model = None
        self.tokenizer = None
        self.base_model = None
        self.is_loaded = False
        self.device = "cpu"
        
    def _check_cuda(self):
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            torch.cuda.empty_cache()
            self.device = "cuda"
            return True
        else:
            print("CUDA not available, using CPU")
            self.device = "cpu"
            return False
    
    def load_model(self):
        if self.is_loaded:
            return True
            
        start_time = time.time()
        print(f"🔄 Diet/Exercise model yükleniyor...")
        
        try:
            self._check_cuda()
            print(f"📱 Cihaz: {self.device}")
            
            base_model_id = "stabilityai/stablelm-2-zephyr-1_6b"
            print(f"📦 Base model: {base_model_id}")
            
            quantization_config = None
            if self.device == "cuda":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=True
                )
                print(f"⚡ CUDA quantization aktif")
            
            print(f"📥 Base model yükleniyor...")
            self.base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                device_map="auto" if self.device == "cuda" else None,
                quantization_config=quantization_config,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                offload_folder="offload" if self.device == "cuda" else None
            )
            print(f"✅ Base model yüklendi")
            
            print(f"📝 Tokenizer yükleniyor...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_id,
                trust_remote_code=True,
                use_fast=True,
                model_max_length=2048
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"✅ Tokenizer yüklendi")
            
            current_dir = os.path.dirname(os.path.abspath(__file__))
            peft_model_id = os.path.join(current_dir, "stablelm-2-zephyr-1_6b")
            print(f"🔍 PEFT model yolu: {peft_model_id}")
            
            if not os.path.exists(peft_model_id):
                print(f"⚠️  PEFT model bulunamadı, base model kullanılıyor")
                self.model = self.base_model
            else:
                print(f"📥 PEFT model yükleniyor...")
                peft_config = PeftConfig.from_pretrained(peft_model_id)
                self.model = PeftModel.from_pretrained(
                    self.base_model,
                    peft_model_id,
                    device_map="auto" if self.device == "cuda" else None,
                    offload_folder="offload" if self.device == "cuda" else None,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
                self.model = self.model.merge_and_unload()
                print(f"✅ PEFT model yüklendi ve merge edildi")
            
            self.model.eval()
            print(f"✅ Model eval moduna alındı")
            
            del self.base_model
            if self.device == "cuda":
                torch.cuda.empty_cache()
                print(f"🧹 CUDA cache temizlendi")
            
            self.is_loaded = True
            load_time = time.time() - start_time
            print(f"✅ Diet/Exercise model başarıyla yüklendi! Süre: {load_time:.2f} saniye")
            return True
            
        except Exception as e:
            print(f"❌ Diet/Exercise model yüklenirken hata: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
            return False
    
    def generate_response_streaming(self, prompt, max_length=512, temperature=0.7, top_p=0.9):
        
        if not self.is_loaded:
            if not self.load_model():
                yield "Model yüklenemedi"
                return
        
        messages = [
            {"role": "system", "content": "You are a registered dietitian and fitness expert. You provide professional diet and exercise advice. IMPORTANT: You must respond ONLY in English. Never use any other language. Provide comprehensive, detailed responses. Always give complete answers with practical advice. Do not use any HTML tags or formatting."},
            {"role": "user", "content": prompt}
        ]
        formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        input_ids = inputs['input_ids'] if 'input_ids' in inputs else inputs.input_ids

        generated_text = ""
        tokens_generated = 0
        
        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=1,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=1.05,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
                
                new_token = outputs[0][-1:]
                
                if new_token.item() == self.tokenizer.eos_token_id:
                    break
                
                new_text = self.tokenizer.decode(new_token, skip_special_tokens=True)
                
                if new_text.strip(): 
                    generated_text += new_text
                    tokens_generated += 1
                    yield new_text
                

                input_ids = outputs

        if tokens_generated == 0:
            yield "I'm sorry, I couldn't generate a response. Please try rephrasing your question."

    def generate_response(self, prompt, max_length=512, temperature=0.7, top_p=0.9):
       
        if not self.is_loaded:
            if not self.load_model():
                return "Model yüklenemedi!"
        
        messages = [
            {"role": "system", "content": "You are a registered dietitian and fitness expert. You provide professional diet and exercise advice. IMPORTANT: You must respond ONLY in English. Never use any other language. Provide comprehensive, detailed responses. Always give complete answers with practical advice. Do not use any HTML tags or formatting."},
            {"role": "user", "content": prompt}
        ]
        formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        input_ids = inputs['input_ids'] if 'input_ids' in inputs else inputs.input_ids
        
        generation_config = {
            "max_new_tokens": max_length,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": 1.05,
            "use_cache": True
        }
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generation_config
            )
        
        response = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        
        # Clean up response - remove HTML tags and special characters
        response = response.replace('<|system|>', '').replace('<|user|>', '').replace('<|assistant|>', '')
        response = response.replace('<', '').replace('>', '').replace('b', '').replace('span', '').replace('style', '').replace('font-size', '').replace('18pt', '').replace(';', '').replace('"', '').replace('=', '').replace('/', '')
        response = response.strip()
        
        if not response or response.isspace():
            return "I'm sorry, I couldn't generate a response. Please try rephrasing your question."
        
        return response

    def generate_response_letter_by_letter(self, prompt, max_length=512, temperature=0.7, top_p=0.9):
        print(f"🔄 generate_response_letter_by_letter başlatılıyor...")
        print(f"📝 Prompt: {prompt[:50]}...")
        print(f"🔢 Max length: {max_length}, Temperature: {temperature}, Top-p: {top_p}")
       
        if not self.is_loaded:
            print(f"📥 Model yükleniyor...")
            if not self.load_model():
                print(f"❌ Model yüklenemedi!")
                yield "Model yüklenemedi!"
                return
        
        print(f"🔄 Chat template uygulanıyor...")
        messages = [
            {"role": "system", "content": "You are a registered dietitian and fitness expert. You provide professional diet and exercise advice. IMPORTANT: You must respond ONLY in English. Never use any other language. Provide comprehensive, detailed responses. Always give complete answers with practical advice. Do not use any HTML tags or formatting."},
            {"role": "user", "content": prompt}
        ]
        formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        print(f"📝 Formatlanmış prompt uzunluğu: {len(formatted_prompt)} karakter")
        
        print(f"🔄 Tokenization yapılıyor...")
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            print(f"📱 Inputs CUDA'ya taşındı")
        
        input_ids = inputs['input_ids'] if 'input_ids' in inputs else inputs.input_ids
        print(f"✅ Tokenization tamamlandı, input shape: {input_ids.shape}")

        generated_text = ""
        tokens_generated = 0
        
        print(f"🔄 Token üretimi başlatılıyor...")
        with torch.no_grad():
            for iteration in range(max_length):
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=1,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=1.05,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
                
                new_token = outputs[0][-1:]
                
                if new_token.item() == self.tokenizer.eos_token_id:
                    print(f"✅ EOS token bulundu, üretim durduruluyor")
                    break
                
                new_text = self.tokenizer.decode(new_token, skip_special_tokens=True)
                
                if new_text.strip():  
                    generated_text += new_text
                    tokens_generated += 1

                    for char in new_text:
                        yield char 
                
                input_ids = outputs
                
                if iteration % 50 == 0 and iteration > 0:
                    print(f"📝 {iteration} iterasyon tamamlandı, {tokens_generated} token üretildi")

        if tokens_generated == 0:
            print(f"❌ Hiç token üretilmedi")
            yield "I'm sorry, I couldn't generate a response. Please try rephrasing your question."
        else:
            print(f"✅ Toplam {tokens_generated} token üretildi")

_model_instance = None

def get_model():
    global _model_instance
    if _model_instance is None:
        _model_instance = DietExerciseModel()
    return _model_instance

def generate_response(prompt, max_length=512, temperature=0.7, top_p=0.9):
    model = get_model()
    return model.generate_response(prompt, max_length, temperature, top_p)

def generate_response_streaming(prompt, max_length=512, temperature=0.7, top_p=0.9):
    model = get_model()
    return model.generate_response_streaming(prompt, max_length, temperature, top_p)

def generate_response_letter_by_letter(prompt, max_length=512, temperature=0.7, top_p=0.9):
    model = get_model()
    return model.generate_response_letter_by_letter(prompt, max_length, temperature, top_p)

def generate_single_response(prompt, max_length=512, temperature=0.7, top_p=0.9):
    """Tek bir soru için yanıt üretir - main_chatbot.py için (gerçek zamanlı streaming)"""
    print(f"🔄 generate_single_response çağrıldı: {prompt[:50]}...")
    
    model = get_model()
    print(f"📱 Model durumu: is_loaded={model.is_loaded}, device={model.device}")
    
    # Model yüklendikten sonra harf harf yanıt üret
    if hasattr(model, 'generate_response_letter_by_letter'):
        print(f"✅ generate_response_letter_by_letter fonksiyonu bulundu")
        char_count = 0
        for char in model.generate_response_letter_by_letter(prompt, max_length, temperature, top_p):
            char_count += 1
            if char_count % 50 == 0:
                print(f"📝 {char_count} karakter üretildi...")
            yield char
        
        if char_count == 0:
            print(f"❌ Hiç karakter üretilmedi")
            yield "I'm sorry, I couldn't generate a response. Please try rephrasing your question."
        else:
            print(f"✅ Toplam {char_count} karakter üretildi")
    else:
        print(f"❌ generate_response_letter_by_letter fonksiyonu bulunamadı")
        yield "I'm sorry, I couldn't generate a response. Please try rephrasing your question."

def interactive_chat():

    model = get_model()
    
    print("\n🤖 Diet & Exercise Chatbot started!")
    print("📝 Letter by letter mode activated!")
    print("=" * 50)
    
    while True:
        user_input = input("👤 You: ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            print("👋 Goodbye!")
            break
        
        if not user_input:
            continue
        
        print("🤖 Bot: ", end="", flush=True)
        
        try:
            response_generated = False
            response_text = ""
            
            for char in model.generate_response_letter_by_letter(user_input):
                print(char, end="", flush=True)
                response_text += char
                response_generated = True
                time.sleep(0.01)  
            
            if not response_generated or not response_text.strip():
                print("I'm sorry, I couldn't generate a response. Please try rephrasing your question.")
            print()
        except Exception as e:
            print("Please try again.")
        
        print()

if __name__ == "__main__":
    interactive_chat() 