import sys
import os
import subprocess
import importlib.util
from pathlib import Path
import torch

# Ana dizini Python path'ine ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# CUDA kontrolü
def check_cuda_availability():
    if torch.cuda.is_available():
        print(" CUDA is available")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        torch.cuda.empty_cache()
        return True
    else:
        print("CUDA not available")
        print("CPU modunda çalışacak (daha yavaş olabilir)")
        return False

# Sentence transformer intent modülünü import et
from sentence_transformer_intent import load_models, predict_intent

class MainChatbot:
    def __init__(self):
        """Ana chatbot sınıfını başlatır"""
        print(" Ana Chatbot başlatılıyor...")
        
        # CUDA durumunu kontrol et
        print(" CUDA durumu kontrol ediliyor...")
        self.cuda_available = check_cuda_availability()
        
        # Niyet tanıma modellerini yükle
        print(" Niyet tanıma modelleri yükleniyor...")
        self.intent_model, self.scaler, self.sentence_model = load_models()
        
        if not all([self.intent_model, self.scaler, self.sentence_model]):
            print(" Niyet tanıma modelleri yüklenemedi!")
            print("Lütfen önce sentence_transformer_intent.py dosyasında model eğitimi yapın.")
            return
        
        print(" Niyet tanıma modelleri yüklendi!")
        
        # Modül yollarını tanımla
        self.modules = {
            "nutrition": {
                "path": "program_prepeare/run.py",
                "name": "Beslenme Planı Modülü",
                "description": "Kişiselleştirilmiş beslenme planları oluşturur"
            },
            "health_rag_info": {
                "path": "rag_info/rag_system.py", 
                "name": "Sağlık Bilgisi Modülü",
                "description": "Hamilelik ve doğum sonrası sağlık bilgileri sağlar"
            },
            "diet_exercise": {
                "path": "empamomodeldeneme/inference.py",
                "name": "Diyet ve Egzersiz Modülü", 
                "description": "Diyet ve egzersiz önerileri sunar"
            },
            "emotional_support": {
                "path": "empamom_emotional_support/inferance.py",
                "name": "Duygusal Destek Modülü",
                "description": "Duygusal destek ve psikolojik yardım sağlar"
            }
        }
        
        print("🚀 Ana Chatbot hazır!")
    
    def predict_user_intent(self, user_message):
        intent, confidence = predict_intent(user_message, self.sentence_model, self.scaler, self.intent_model)
        return intent, confidence
    
    def run_nutrition_module(self, user_message):
        print(f"\n🥗 {self.modules['nutrition']['name']} başlatılıyor...")
        print(f"📝 Kullanıcı mesajı: {user_message}")
        
        try:
            # program_prepeare klasörünü Python path'ine ekle
            nutrition_dir = os.path.join(os.path.dirname(__file__), "program_prepeare")
            sys.path.insert(0, nutrition_dir)
            
            # advanced_nutrition_generator modülünü import et
            import advanced_nutrition_generator
            
            # Modülü çalıştır
            print("🔄 Beslenme modülü başlatılıyor...")
            generator = advanced_nutrition_generator.AdvancedNutritionGenerator()
            generator.generate_program()
            
            # Path'i geri al
            sys.path.pop(0)
                
        except ImportError as e:
            print(f"❌ Beslenme modülü import hatası: {e}")
            print("💡 program_prepeare klasöründeki requirements.txt'yi yüklemeyi deneyin:")
            print("   cd program_prepeare")
            print("   pip install -r requirements.txt")
        except Exception as e:
            print(f"❌ Beslenme modülü çalıştırılırken hata: {e}")
            print("💡 Alternatif çözüm: program_prepeare klasöründe ayrıca çalıştırın:")
            print("   cd program_prepeare")
            print("   python run.py")
    
    def run_health_rag_module(self, user_message):
        """Sağlık RAG modülünü çalıştırır"""
        print(f"\n🏥 {self.modules['health_rag_info']['name']} başlatılıyor...")
        print(f"📝 Kullanıcı mesajı: {user_message}")
        
        try:
            # RAG sistemini import et
            spec = importlib.util.spec_from_file_location(
                "rag_system", 
                "rag_info/rag_system.py"
            )
            rag_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(rag_module)
            
            # RAG sistemini başlat ve streaming yanıt üret
            rag_system = rag_module.RAGSystem()
            
            print(" RAG Yanıtı:")
            print("💬 ", end="", flush=True)
            
            # Streaming yanıt üret
            response_generated = False
            response_text = ""
            
            # Benzer chunk'ları ara
            similar_chunks = rag_system.search_similar_chunks(user_message, top_k=5)
            context = rag_system.create_context(similar_chunks)
            
            # Streaming yanıt üret
            for token in rag_system.generate_response_streaming(user_message, context):
                print(token, end="", flush=True)
                response_text += token
                response_generated = True
                import time
                time.sleep(0.01)  # Harf harf yazma hızı
            
            if not response_generated:
                print("I'm sorry, I couldn't generate a response. Please try rephrasing your question.")
            print()
                
        except Exception as e:
            print(f"AG modülü çalıştırılırken hata: {e}")
    
    def run_diet_exercise_module(self, user_message):
        """Diyet ve egzersiz modülünü çalıştırır"""
        print(f"\n {self.modules['diet_exercise']['name']} başlatılıyor...")
        print(f" Kullanıcı mesajı: {user_message}")
        
        try:
            # Inference modülünü import et
            spec = importlib.util.spec_from_file_location(
                "inference", 
                "empamomodeldeneme/inference.py"
            )
            inference_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(inference_module)
            
            # Inference.py'nin kendi fonksiyonunu kullan
            if hasattr(inference_module, 'generate_single_response'):
                print("🔄 Diyet/Egzersiz modeli yanıt üretiyor...")
                print("⏳ Model yükleniyor (ilk kullanımda biraz zaman alabilir)...")
                print("🤖 Model Yanıtı:")
                print("💬 ", end="", flush=True)
                
                # Inference.py'nin kendi fonksiyonunu kullanarak gerçek zamanlı yanıt al
                response_generated = False
                for char in inference_module.generate_single_response(user_message):
                    print(char, end="", flush=True)
                    response_generated = True
                    import time
                    time.sleep(0.01)
                
                if not response_generated:
                    print("I'm sorry, I couldn't generate a response. Please try rephrasing your question.")
                print()
            else:
                print("❌ Inference modülünde generate_single_response fonksiyonu bulunamadı!")
                
        except Exception as e:
            print(f"❌ Diyet/Egzersiz modülü çalıştırılırken hata: {e}")
            print("💡 Hata detayı:", str(e))
            print("🔧 Çözüm önerileri:")
            print("   1. CUDA sürücülerinin güncel olduğundan emin olun")
            print("   2. PyTorch'un CUDA versiyonunun yüklü olduğunu kontrol edin")
            print("   3. Model dosyalarının doğru konumda olduğunu kontrol edin")
            print("   4. Gerekli kütüphanelerin yüklü olduğunu kontrol edin:")
            print("      pip install torch transformers peft bitsandbytes")
    
    def run_emotional_support_module(self, user_message):
        """Duygusal destek modülünü çalıştırır"""
        print(f"\n💙 {self.modules['emotional_support']['name']} başlatılıyor...")
        print(f"📝 Kullanıcı mesajı: {user_message}")
        
        try:
            # Emotional support modülünü import et
            spec = importlib.util.spec_from_file_location(
                "emotional_inference", 
                "empamom_emotional_support/inferance.py"
            )
            emotional_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(emotional_module)
            
            # Emotional support modülünün kendi fonksiyonunu kullan
            if hasattr(emotional_module, 'generate_single_response'):
                print("🔄 Duygusal Destek modeli yanıt üretiyor...")
                print("⏳ Model yükleniyor (ilk kullanımda biraz zaman alabilir)...")
                print("🤖 Duygusal Destek Yanıtı:")
                print("💬 ", end="", flush=True)
                
                # Emotional support modülünün kendi fonksiyonunu kullanarak gerçek zamanlı yanıt al
                response_generated = False
                for char in emotional_module.generate_single_response(user_message):
                    print(char, end="", flush=True)
                    response_generated = True
                    import time
                    time.sleep(0.01)
                
                if not response_generated:
                    print("I'm sorry, I couldn't generate a response. Please try rephrasing your question.")
                print()
            else:
                print("❌ Emotional support modülünde generate_single_response fonksiyonu bulunamadı!")
                
        except Exception as e:
            print(f"❌ Duygusal Destek modülü çalıştırılırken hata: {e}")
            print("💡 Hata detayı:", str(e))
            print("🔧 Çözüm önerileri:")
            print("   1. CUDA sürücülerinin güncel olduğundan emin olun")
            print("   2. PyTorch'un CUDA versiyonunun yüklü olduğunu kontrol edin")
            print("   3. Model dosyalarının doğru konumda olduğunu kontrol edin")
            print("   4. Gerekli kütüphanelerin yüklü olduğunu kontrol edin:")
            print("      pip install torch transformers peft bitsandbytes")
    
    def process_user_message(self, user_message):
        """Kullanıcı mesajını işler ve uygun modülü çalıştırır"""
        print(f"\n🔍 Mesaj analiz ediliyor: '{user_message}'")
        
        # Niyet tahmini yap
        intent, confidence = self.predict_user_intent(user_message)
        
        print(f"🎯 Tahmin edilen kategori: {intent}")
        print(f"📊 Güven skoru: {confidence:.3f} ({confidence*100:.1f}%)")
        
        # Güven skoru kontrolü
        if confidence < 0.6:
            print("⚠️  Düşük güven skoru - Mesaj anlaşılamadı")
            print("💡 Lütfen mesajınızı daha açık bir şekilde ifade edin.")
            return
        
        # Kategoriye göre modül çalıştır
        if intent == "nutrition":
            self.run_nutrition_module(user_message)
            
        elif intent == "health_rag_info":
            self.run_health_rag_module(user_message)
            
        elif intent == "diet_exercise":
            self.run_diet_exercise_module(user_message)
            
        elif intent == "emotional_support":
            self.run_emotional_support_module(user_message)
            
        elif intent == "anlasilamadi":
            print("❓ Mesajınız anlaşılamadı.")
            print("💡 Lütfen mesajınızı daha açık bir şekilde ifade edin.")
            
        else:
            print(f"❓ Bilinmeyen kategori: {intent}")
    
    def show_help(self):
        """Yardım menüsünü gösterir"""
        print("\n" + "="*60)
        print("🤖 ANA CHATBOT - YARDIM MENÜSÜ")
        print("="*60)
        print("Bu chatbot şu kategorilerde size yardımcı olabilir:")
        
        for category, info in self.modules.items():
            print(f"\n📋 {info['name']} ({category}):")
            print(f"   {info['description']}")
        
        print("\n💡 Örnek mesajlar:")
        examples = {
            "nutrition": [
                "Can you create a meal plan for breastfeeding?",
                "What should I eat to boost my milk supply?",
                "I need a diet plan for postpartum recovery"
            ],
            "health_rag_info": [
                "What are the symptoms of postpartum depression?",
                "How can I manage my baby's fever?",
                "Is it normal to have back pain during pregnancy?"
            ],
            "diet_exercise": [
                "What exercises are safe during pregnancy?",
                "How can I lose weight after childbirth?",
                "What should I eat to stay healthy?"
            ],
            "emotional_support": [
                "I'm feeling overwhelmed with motherhood",
                "I'm struggling with postpartum depression",
                "I need emotional support during pregnancy"
            ]
        }
        
        for category, msgs in examples.items():
            print(f"\n{self.modules[category]['name']}:")
            for msg in msgs:
                print(f"  • {msg}")
        
        print("\n" + "="*60)
    
    def interactive_chat(self):
        """Etkileşimli sohbet modu"""
        print("\n" + "="*60)
        print("🤖 ANA CHATBOT - ETKİLEŞİMLİ MOD")
        print("="*60)
        print("Komutlar:")
        print("  'help' - Yardım menüsünü göster")
        print("  'quit' veya 'exit' - Çıkış yap")
        print("  'clear' - Ekranı temizle")
        print("="*60)
        
        while True:
            try:
                user_input = input("\n👤 Siz: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'çık', 'çıkış']:
                    print("👋 Görüşürüz!")
                    break
                
                if user_input.lower() == 'help':
                    self.show_help()
                    continue
                
                if user_input.lower() == 'clear':
                    os.system('cls' if os.name == 'nt' else 'clear')
                    continue
                
                if not user_input:
                    print("💡 Lütfen bir mesaj yazın.")
                    continue
                
                # Mesajı işle
                self.process_user_message(user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 Görüşürüz!")
                break
            except Exception as e:
                print(f"❌ Hata oluştu: {e}")
                continue

def main():
    """Ana fonksiyon"""
    print("🚀 Ana Chatbot başlatılıyor...")
    
    # Chatbot'u başlat
    chatbot = MainChatbot()
    
    # Etkileşimli moda geç
    chatbot.interactive_chat()

if __name__ == "__main__":
    main() 