import sys
import os
import subprocess
import importlib.util
from pathlib import Path
import torch
import json
import uuid
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import threading
import time

# Ana dizini Python path'ine ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# CUDA kontrolü
def check_cuda_availability():
    if torch.cuda.is_available():
        print("✅ CUDA is available")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        torch.cuda.empty_cache()
        return True
    else:
        print("❌ CUDA not available")
        print("CPU modunda çalışacak (daha yavaş olabilir)")
        return False

# Sentence transformer intent modülünü import et
from sentence_transformer_intent import load_models, predict_intent

# PDF generator modülünü import et
from pdf_generator import PDFGenerator

app = Flask(__name__)
CORS(app)  # Flutter uygulamasından gelen isteklere izin ver

class APIChatbot:
    def __init__(self):
        """API chatbot sınıfını başlatır"""
        print("🚀 API Chatbot başlatılıyor...")
        
        # CUDA durumunu kontrol et
        print("🔍 CUDA durumu kontrol ediliyor...")
        self.cuda_available = check_cuda_availability()
        
        # Niyet tanıma modellerini yükle
        print("📚 Niyet tanıma modelleri yükleniyor...")
        self.intent_model, self.scaler, self.sentence_model = load_models()
        
        if not all([self.intent_model, self.scaler, self.sentence_model]):
            print("❌ Niyet tanıma modelleri yüklenemedi!")
            print("Lütfen önce sentence_transformer_intent.py dosyasında model eğitimi yapın.")
            return
        
        print("✅ Niyet tanıma modelleri yüklendi!")
        
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
        
        # Nutrition session'ları için storage
        self.nutrition_sessions = {}
        
        # PDF generator'ı başlat
        self.pdf_generator = PDFGenerator()
        
        # Nutrition modülü için soru sırası
        self.nutrition_questions = [
            ("age", "What is your age?", "int"),
            ("weight", "What is your current weight (in kg)?", "float"),
            ("height", "What is your height (in cm)?", "float"),
            ("activity_level", "What is your activity level? (sedentary/light/moderate/active/very_active)", "str"),
            ("occupation", "What is your occupation? (desk_job/physical_work/mixed)", "str"),
            ("sleep_hours", "How many hours do you sleep per night?", "int"),
            ("stress_level", "What is your stress level? (low/medium/high)", "str"),
            ("pregnancy_status", "Are you currently pregnant? (yes/no)", "str"),
            ("pregnancy_weeks", "How many weeks pregnant are you?", "int"),
            ("postpartum_weeks", "How many weeks ago did you give birth?", "int"),
            ("delivery_method", "What was your delivery method? (vaginal/caesarean/assisted)", "str"),
            ("breastfeeding", "Are you breastfeeding? (yes/no)", "str"),
            ("breastfeeding_weeks", "If breastfeeding, how many weeks have you been breastfeeding?", "int"),
            ("allergies", "Do you have any food allergies? (list them or 'none')", "str"),
            ("medical_conditions", "Do you have any medical conditions? (diabetes/heart_disease/hypertension/none)", "str"),
            ("medications", "Are you taking any medications? (list them or 'none')", "str"),
            ("digestive_issues", "Do you have any digestive issues? (ibs/acid_reflux/none)", "str"),
            ("dietary_preferences", "Any dietary preferences? (vegetarian/vegan/keto/paleo/mediterranean/none)", "str"),
            ("food_dislikes", "Any foods you dislike or avoid? (list them or 'none')", "str"),
            ("cooking_skills", "What are your cooking skills? (beginner/intermediate/advanced)", "str"),
            ("goals", "What are your nutrition goals? (weight_loss/weight_gain/maintenance/health_improvement/energy_boost)", "str"),
            ("program_duration", "How many days would you like the program for? (default: 1)", "int"),
            ("meals_per_day", "How many meals per day do you prefer? (3/4/5/6)", "int"),
            ("snacks", "Do you want snacks included? (yes/no)", "str"),
            ("cooking_time", "How much time can you spend cooking per day? (quick/medium/elaborate)", "str"),
            ("budget", "What's your food budget level? (low/medium/high)", "str"),
            ("cuisine_preference", "Any cuisine preferences? (mediterranean/asian/italian/mexican/none)", "str"),
            ("spice_tolerance", "What's your spice tolerance? (mild/medium/hot)", "str"),
            ("meal_prep", "Do you prefer meal prep or daily cooking? (meal_prep/daily_cooking)", "str")
        ]
        
        print("🚀 API Chatbot hazır!")
    
    def predict_user_intent(self, user_message):
        intent, confidence = predict_intent(user_message, self.sentence_model, self.scaler, self.intent_model)
        return intent, confidence
    
    def create_nutrition_session(self):
        """Yeni bir nutrition session oluşturur"""
        session_id = str(uuid.uuid4())
        self.nutrition_sessions[session_id] = {
            "current_question_index": 0,
            "user_data": {},
            "created_at": datetime.now(),
            "last_activity": datetime.now()
        }
        return session_id
    
    def get_next_nutrition_question(self, session_id):
        """Nutrition session'ı için bir sonraki soruyu döndürür"""
        if session_id not in self.nutrition_sessions:
            return None
        
        session = self.nutrition_sessions[session_id]
        session["last_activity"] = datetime.now()
        
        if session["current_question_index"] >= len(self.nutrition_questions):
            return None
        
        field, question, data_type = self.nutrition_questions[session["current_question_index"]]
        
        return {
            "field": field,
            "question": question,
            "data_type": data_type,
            "question_number": session["current_question_index"] + 1,
            "total_questions": len(self.nutrition_questions)
        }
    
    def answer_nutrition_question(self, session_id, answer):
        """Nutrition session'ı için verilen cevabı kaydeder"""
        if session_id not in self.nutrition_sessions:
            return False, "Session bulunamadı"
        
        session = self.nutrition_sessions[session_id]
        session["last_activity"] = datetime.now()
        
        if session["current_question_index"] >= len(self.nutrition_questions):
            return False, "Tüm sorular zaten tamamlandı"
        
        field, question, data_type = self.nutrition_questions[session["current_question_index"]]
        
        # Veri tipini kontrol et ve dönüştür
        try:
            if data_type == "int":
                session["user_data"][field] = int(answer)
            elif data_type == "float":
                session["user_data"][field] = float(answer)
            else:
                session["user_data"][field] = answer
        except ValueError:
            return False, f"Geçersiz veri tipi. {data_type} bekleniyor."
        
        session["current_question_index"] += 1
        
        return True, "Cevap kaydedildi"
    
    def generate_nutrition_program(self, session_id, user_id=None, username=None):
        """Nutrition programını oluşturur ve PDF'e kaydeder"""
        if session_id not in self.nutrition_sessions:
            return False, "Session bulunamadı"
        
        session = self.nutrition_sessions[session_id]
        
        if session["current_question_index"] < len(self.nutrition_questions):
            return False, "Tüm sorular henüz tamamlanmadı"
        
        try:
            # Nutrition modülünü import et ve çalıştır
            nutrition_dir = os.path.join(os.path.dirname(__file__), "program_prepeare")
            sys.path.insert(0, nutrition_dir)
            
            import advanced_nutrition_generator
            
            # Modülü çalıştır
            generator = advanced_nutrition_generator.AdvancedNutritionGenerator()
            
            # User data'yı set et
            generator.user_data = session["user_data"]
            
            # Program oluştur
            health_analysis = generator._analyze_health_data(session["user_data"])
            prompt = generator._create_comprehensive_prompt(session["user_data"], health_analysis)
            program_content = generator._call_gemini_api(prompt)
            
            # PDF oluştur ve veritabanına kaydet
            if user_id and username:
                pdf_result = self.pdf_generator.create_nutrition_pdf(
                    user_id, username, program_content, session["user_data"]
                )
                
                if pdf_result["success"]:
                    result = {
                        "success": True,
                        "program_content": program_content,
                        "pdf_info": pdf_result
                    }
                else:
                    result = {
                        "success": True,
                        "program_content": program_content,
                        "pdf_error": pdf_result["message"]
                    }
            else:
                result = {
                    "success": True,
                    "program_content": program_content
                }
            
            # Session'ı temizle
            del self.nutrition_sessions[session_id]
            
            return True, result
            
        except Exception as e:
            return False, f"Program oluşturulurken hata: {str(e)}"
        finally:
            # Path'i geri al
            if nutrition_dir in sys.path:
                sys.path.remove(nutrition_dir)
    
    def run_health_rag_module(self, user_message):
        """Sağlık RAG modülünü çalıştırır"""
        try:
            # RAG sistemini import et
            spec = importlib.util.spec_from_file_location(
                "rag_system", 
                "rag_info/rag_system.py"
            )
            rag_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(rag_module)
            
            # RAG sistemini başlat
            rag_system = rag_module.RAGSystem()
            
            # Benzer chunk'ları ara
            similar_chunks = rag_system.search_similar_chunks(user_message, top_k=5)
            context = rag_system.create_context(similar_chunks)
            
            # Yanıt üret
            response = rag_system.generate_response(user_message, context)
            
            # Generator ise string'e çevir
            if hasattr(response, '__iter__') and not isinstance(response, str):
                response = ''.join(response)
            
            return True, response
                
        except Exception as e:
            return False, f"RAG modülü çalıştırılırken hata: {str(e)}"
    
    def run_diet_exercise_module(self, user_message):
        """Diyet ve egzersiz modülünü çalıştırır"""
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
                response = inference_module.generate_single_response(user_message)
                
                # Generator ise string'e çevir
                if hasattr(response, '__iter__') and not isinstance(response, str):
                    response = ''.join(response)
                
                # System tag'lerini ve diğer karakterleri temizle
                response = response.replace('<|system|>', '').replace('<|user|>', '').replace('<|assistant|>', '')
                response = response.replace('<', '').replace('>', '').replace('b', '').replace('span', '').replace('style', '').replace('font-size', '').replace('18pt', '').replace(';', '').replace('"', '').replace('=', '').replace('/', '')
                response = response.strip()
                
                return True, response
            else:
                return False, "Inference modülünde generate_single_response fonksiyonu bulunamadı!"
                
        except Exception as e:
            return False, f"Diyet/Egzersiz modülü çalıştırılırken hata: {str(e)}"
    
    def run_emotional_support_module(self, user_message):
        """Duygusal destek modülünü çalıştırır"""
        try:
            print(f"🔍 Duygusal destek modülü başlatılıyor...")
            
            # Emotional support modülünü import et
            spec = importlib.util.spec_from_file_location(
                "emotional_inference", 
                "empamom_emotional_support/inferance.py"
            )
            emotional_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(emotional_module)
            
            print(f"✅ Emotional support modülü başarıyla import edildi")
            
            # Emotional support modülünün kendi fonksiyonunu kullan
            if hasattr(emotional_module, 'generate_single_response'):
                print(f"🔄 Duygusal destek yanıtı üretiliyor...")
                
                # generate_single_response bir generator olduğu için tüm karakterleri topla
                response_chars = []
                char_count = 0
                for char in emotional_module.generate_single_response(user_message):
                    response_chars.append(char)
                    char_count += 1
                    if char_count % 50 == 0:  # Her 50 karakterde bir log
                        print(f"📝 {char_count} karakter üretildi...")
                
                response = ''.join(response_chars)
                print(f"✅ Toplam {len(response_chars)} karakter üretildi")
                
                # System tag'lerini ve diğer karakterleri temizle
                response = response.replace('<|system|>', '').replace('<|user|>', '').replace('<|assistant|>', '')
                response = response.replace('<', '').replace('>', '').replace('b', '').replace('span', '').replace('style', '').replace('font-size', '').replace('18pt', '').replace(';', '').replace('"', '').replace('=', '').replace('/', '')
                response = response.strip()
                
                print(f"📝 Temizlenmiş yanıt uzunluğu: {len(response)}")
                
                # Boş yanıt kontrolü
                if not response or response == "I'm sorry, I couldn't generate a response. Please try rephrasing your question.":
                    print(f"❌ Boş yanıt üretildi")
                    return False, "Duygusal destek modülü yanıt üretemedi. Lütfen mesajınızı tekrar deneyin."
                
                print(f"✅ Duygusal destek yanıtı başarıyla üretildi")
                return True, response
            else:
                print(f"❌ generate_single_response fonksiyonu bulunamadı")
                return False, "Emotional support modülünde generate_single_response fonksiyonu bulunamadı!"
                
        except Exception as e:
            print(f"❌ Duygusal destek modülü hatası: {str(e)}")
            import traceback
            traceback.print_exc()
            return False, f"Duygusal Destek modülü çalıştırılırken hata: {str(e)}"
    
    def process_user_message(self, user_message):
        """Kullanıcı mesajını işler ve uygun modülü çalıştırır"""
        # Niyet tahmini yap
        intent, confidence = self.predict_user_intent(user_message)
        
        # Güven skoru kontrolü
        if confidence < 0.6:
            return {
                "success": False,
                "message": "Mesajınız anlaşılamadı. Lütfen mesajınızı daha açık bir şekilde ifade edin.",
                "confidence": confidence
            }
        
        # Kategoriye göre modül çalıştır
        if intent == "nutrition":
            # Nutrition için session oluştur
            session_id = self.create_nutrition_session()
            next_question = self.get_next_nutrition_question(session_id)
            
            return {
                "success": True,
                "intent": intent,
                "confidence": confidence,
                "session_id": session_id,
                "next_question": next_question,
                "message": "Beslenme planı oluşturmak için size bazı sorular soracağım. İlk soru: " + next_question["question"]
            }
            
        elif intent == "health_rag_info":
            success, response = self.run_health_rag_module(user_message)
            return {
                "success": success,
                "intent": intent,
                "confidence": confidence,
                "message": response
            }
            
        elif intent == "diet_exercise":
            success, response = self.run_diet_exercise_module(user_message)
            return {
                "success": success,
                "intent": intent,
                "confidence": confidence,
                "message": response
            }
            
        elif intent == "emotional_support":
            success, response = self.run_emotional_support_module(user_message)
            return {
                "success": success,
                "intent": intent,
                "confidence": confidence,
                "message": response
            }
            
        elif intent == "anlasilamadi":
            return {
                "success": False,
                "intent": intent,
                "confidence": confidence,
                "message": "Mesajınız anlaşılamadı. Lütfen mesajınızı daha açık bir şekilde ifade edin."
            }
            
        else:
            return {
                "success": False,
                "intent": intent,
                "confidence": confidence,
                "message": f"Bilinmeyen kategori: {intent}"
            }
    
    def cleanup_expired_sessions(self):
        """Süresi dolmuş session'ları temizler"""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session in self.nutrition_sessions.items():
            if current_time - session["last_activity"] > timedelta(hours=1):
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.nutrition_sessions[session_id]

# Global chatbot instance
chatbot = APIChatbot()

@app.route('/api/chat', methods=['POST'])
def chat():
    """Ana chat endpoint'i"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({
                "success": False,
                "message": "Mesaj boş olamaz"
            }), 400
        
        # Session temizliği
        chatbot.cleanup_expired_sessions()
        
        # Mesajı işle
        result = chatbot.process_user_message(user_message)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Sunucu hatası: {str(e)}"
        }), 500

@app.route('/api/nutrition/answer', methods=['POST'])
def nutrition_answer():
    """Nutrition sorularına cevap endpoint'i"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        answer = data.get('answer', '').strip()
        user_id = data.get('user_id')
        username = data.get('username')
        
        if not session_id or not answer:
            return jsonify({
                "success": False,
                "message": "Session ID ve cevap gerekli"
            }), 400
        
        # Cevabı kaydet
        success, message = chatbot.answer_nutrition_question(session_id, answer)
        
        if not success:
            return jsonify({
                "success": False,
                "message": message
            }), 400
        
        # Bir sonraki soruyu al
        next_question = chatbot.get_next_nutrition_question(session_id)
        
        if next_question is None:
            # Tüm sorular tamamlandı, program oluştur
            success, result = chatbot.generate_nutrition_program(session_id, user_id, username)
            
            if success:
                return jsonify({
                    "success": True,
                    "message": result["program_content"],
                    "pdf_info": result.get("pdf_info"),
                    "pdf_error": result.get("pdf_error"),
                    "completed": True
                })
            else:
                return jsonify({
                    "success": False,
                    "message": result,
                    "completed": True
                })
        else:
            return jsonify({
                "success": True,
                "next_question": next_question,
                "message": "Cevap kaydedildi. Sıradaki soru: " + next_question["question"],
                "completed": False
            })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Sunucu hatası: {str(e)}"
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Sağlık kontrolü endpoint'i"""
    return jsonify({
        "status": "healthy",
        "cuda_available": chatbot.cuda_available,
        "models_loaded": all([chatbot.intent_model, chatbot.scaler, chatbot.sentence_model]),
        "active_sessions": len(chatbot.nutrition_sessions)
    })

@app.route('/api/modules', methods=['GET'])
def get_modules():
    """Mevcut modülleri listeler"""
    return jsonify({
        "modules": chatbot.modules
    })

@app.route('/api/programs/<user_id>', methods=['GET'])
def get_user_programs(user_id):
    """Kullanıcının programlarını listeler"""
    try:
        programs = chatbot.pdf_generator.get_user_programs(user_id)
        return jsonify({
            "success": True,
            "programs": programs
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Programlar getirilirken hata: {str(e)}"
        }), 500

@app.route('/api/programs/<program_id>/download', methods=['GET'])
def download_program(program_id):
    """Program PDF'ini indirir"""
    try:
        file_path = chatbot.pdf_generator.get_program_file_path(program_id)
        
        if not file_path or not os.path.exists(file_path):
            return jsonify({
                "success": False,
                "message": "PDF dosyası bulunamadı"
            }), 404
        
        return send_file(file_path, as_attachment=True)
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"PDF indirilirken hata: {str(e)}"
        }), 500

@app.route('/api/programs/<program_id>/view', methods=['GET'])
def view_program(program_id):
    """Program PDF'ini görüntüler"""
    try:
        file_path = chatbot.pdf_generator.get_program_file_path(program_id)
        
        if not file_path or not os.path.exists(file_path):
            return jsonify({
                "success": False,
                "message": "PDF dosyası bulunamadı"
            }), 404
        
        return send_file(file_path, mimetype='application/pdf')
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"PDF görüntülenirken hata: {str(e)}"
        }), 500

@app.route('/api/programs/<program_id>', methods=['DELETE'])
def delete_program(program_id):
    """Programı siler"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        
        if not user_id:
            return jsonify({
                "success": False,
                "message": "User ID gerekli"
            }), 400
        
        success = chatbot.pdf_generator.delete_program(program_id, user_id)
        
        if success:
            return jsonify({
                "success": True,
                "message": "Program başarıyla silindi"
            })
        else:
            return jsonify({
                "success": False,
                "message": "Program silinemedi"
            }), 404
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Program silinirken hata: {str(e)}"
        }), 500

if __name__ == '__main__':
    print("🚀 API Chatbot sunucusu başlatılıyor...")
    print("📡 Sunucu http://localhost:5000 adresinde çalışacak")
    print("🔗 Endpoint'ler:")
    print("   POST /api/chat - Ana chat")
    print("   POST /api/nutrition/answer - Nutrition sorularına cevap")
    print("   GET  /api/health - Sağlık kontrolü")
    print("   GET  /api/modules - Modül listesi")
    print("   GET  /api/programs/<user_id> - Kullanıcı programlarını listele")
    print("   GET  /api/programs/<program_id>/download - Program PDF'ini indir")
    print("   GET  /api/programs/<program_id>/view - Program PDF'ini görüntüle")
    print("   DELETE /api/programs/<program_id> - Programı sil")
    
    app.run(host='0.0.0.0', port=5000, debug=False) 