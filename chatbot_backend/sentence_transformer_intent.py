import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os
import json
from sklearn.preprocessing import StandardScaler

def load_and_combine_data(file_path1="maternal_health_data.csv", file_path2="Expanded_Maternal_Health_Support_Dataset__v2_.csv"):
    try:
        # İlk dosyayı yükle
        df1 = pd.read_csv(file_path1)
        print(f"İlk dosya yüklendi: {len(df1)} örnek")
        
        # İkinci dosyayı yükle
        df2 = pd.read_csv(file_path2)
        print(f"İkinci dosya yüklendi: {len(df2)} örnek")
        
        # Verileri birleştir
        combined_df = pd.concat([df1, df2], ignore_index=True)
        print(f"Birleştirilmiş veri seti: {len(combined_df)} örnek")
        
        # Kategori dağılımını göster
        print("\nKategori dağılımı:")
        print(combined_df['category'].value_counts())
        
        # Boş değerleri kontrol et
        print(f"\nBoş değerler:")
        print(combined_df.isnull().sum())
        
        # Boş değerleri temizle
        combined_df = combined_df.dropna()
        print(f"Temizlenmiş veri seti: {len(combined_df)} örnek")
        
        return combined_df
        
    except FileNotFoundError as e:
        print(f"Hata: Dosya bulunamadı - {e}")
        return None
    except Exception as e:
        print(f"Hata: Veri yükleme sırasında bir sorun oluştu - {e}")
        return None

def prepare_data(df):
    """Veriyi model eğitimi için hazırlar."""
    texts = df['user_message'].tolist()
    labels = df['category'].tolist()
    
    # Metinleri temizle 
    cleaned_texts = []
    for text in texts:
        if isinstance(text, str):
            cleaned_text = ' '.join(text.split())
            cleaned_texts.append(cleaned_text)
        else:
            cleaned_texts.append(str(text))
    
    return cleaned_texts, labels

def create_embeddings(texts, model_name='all-MiniLM-L6-v2'):
       
    # Modeli yükle
    model = SentenceTransformer(model_name)
    
    print("Embedding'ler oluşturuluyor...")
    # Metinleri embedding'lere dönüştür
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    
    return embeddings, model

def train_and_save_model(data_path1="maternal_health_data.csv", 
                        data_path2="Expanded_Maternal_Health_Support_Dataset__v2_.csv", 
                        model_dir="sentence_transformer_models"):
    

    
    # Verileri yükle ve birleştir
    combined_df = load_and_combine_data(data_path1, data_path2)
    
    if combined_df is None or len(combined_df) == 0:
        print("Hata: Veri kümesi yüklenemedi veya boş.")
        return
    
    print("2. Veriler hazırlanıyor...")
    texts, labels = prepare_data(combined_df)
    
    if not texts or not labels:
        print("Hata: Veri kümesi boş veya hazırlanamadı.")
        return
    
    print(f"Toplam örnek sayısı: {len(texts)}")
    print(f"Benzersiz kategoriler: {len(set(labels))}")
    
    
    X, sentence_model = create_embeddings(texts)
    y = labels
    
    print(f"Embedding boyutu: {X.shape}")
    
    # Veriyi eğitim ve test setlerine ayırma
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Eğitim seti: {X_train.shape[0]} örnek")
    print(f"Test seti: {X_test.shape[0]} örnek")
    
    # Veriyi normalize et 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # SVM modelini eğitme
    model = SVC(
        kernel='rbf', 
        probability=True,  
        random_state=42,
        C=1.0,  
        gamma='scale',  
        class_weight='balanced'  
    )
    
    model.fit(X_train_scaled, y_train)
    print("Model eğitimi tamamlandı.")
    
    # Model performansını değerlendirme
    y_pred = model.predict(X_test_scaled)
    
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Kategoriler:", model.classes_)
    print(cm)
    
    # Modelleri kaydetme
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "intent_model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    sentence_model_path = os.path.join(model_dir, "sentence_transformer")
    
    # SVM modelini kaydet
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Scaler'ı kaydet
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    #Sentence Transformer modelini kaydet
    sentence_model.save(sentence_model_path)
    
    print(f"SVM modeli kaydedildi: {model_path}")
    print(f"Scaler kaydedildi: {scaler_path}")
    print(f"Sentence Transformer modeli kaydedildi: {sentence_model_path}")
    
    #Model bilgilerini kaydet
    model_info = {
        'categories': model.classes_.tolist(),
        'embedding_dimension': X.shape[1],
        'n_samples': len(texts),
        'training_samples': X_train.shape[0],
        'test_samples': X_test.shape[0],
        'sentence_transformer_model': 'all-MiniLM-L6-v2'
    }
    
    info_path = os.path.join(model_dir, "model_info.json")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)
    
    print(f"Model bilgileri kaydedildi: {info_path}")

def load_models(model_dir="sentence_transformer_models"):
    model_path = os.path.join(model_dir, "intent_model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    sentence_model_path = os.path.join(model_dir, "sentence_transformer")
    
    if not all(os.path.exists(path) for path in [model_path, scaler_path, sentence_model_path]):
        print(f"Hata: Gerekli model dosyaları '{model_dir}' dizininde bulunamadı.")

        return None, None, None
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Sentence Transformer modelini yükle
        sentence_model = SentenceTransformer(sentence_model_path)
        
        print("Tüm modeller başarıyla yüklendi.")
        return model, scaler, sentence_model
        
    except Exception as e:
        print(f"Hata: Model yüklenirken bir sorun oluştu - {e}")
        return None, None, None

def predict_intent(user_query, sentence_model, scaler, model, confidence_threshold=0.6):

    if not all([sentence_model, scaler, model]):
        return "modeller_yuklenemedi", 0.0
    
    try:
        # Metni temizle
        if isinstance(user_query, str):
            cleaned_query = ' '.join(user_query.split())
        else:
            cleaned_query = str(user_query)
        
        # Sentence Transformer ile embedding oluştur
        query_embedding = sentence_model.encode([cleaned_query])
        
        # Embedding'i normalize et
        query_embedding_scaled = scaler.transform(query_embedding)
        
        # Tahmin olasılıklarını al
        probabilities = model.predict_proba(query_embedding_scaled)[0]
        
        # En yüksek olasılıklı niyeti ve güven skorunu bul
        max_prob_idx = np.argmax(probabilities)
        max_prob = probabilities[max_prob_idx]
        predicted_intent = model.classes_[max_prob_idx]
        
        # Güven eşiği kontrolü
        if max_prob < confidence_threshold:
            return "anlasilamadi", max_prob
        
        return predicted_intent, max_prob
        
    except Exception as e:
        print(f"Hata: Tahmin sırasında bir sorun oluştu - {e}")
        return "hata_olustu", 0.0

#def get_intent_examples():
#    examples = {
#        "health_rag_info": [
#            "What are the symptoms of postpartum depression?",
#            "How can I manage my baby's fever?",
#            "Is it normal to have back pain during pregnancy?",
#            "What should I do if my baby is not sleeping well?",
#            "I am 6 months pregnant and feel pain in my vagina is it normal?",
#            "What causes vaginal bleeding during pregnancy?"
#        ],
#        "nutrition": [
#            "Can you create a meal plan for breastfeeding?",
#            "What should I eat to boost my milk supply?",
#            "I need a diet plan for postpartum recovery",
#            "What foods are good for pregnancy?",
#            "Can you help me create a personalized meal plan?",
#            "What should I eat to support breastfeeding?"
#        ],
#        "emotional_support": [
#            "I'm feeling overwhelmed as a new mom",
#            "How can I cope with postpartum depression?",
#            "I feel anxious about being a good mother",
#            "I'm struggling with the emotional changes",
#            "I feel isolated and alone",
#            "I feel like I'm not a good enough mother"
#        ],
#        "diet_exercise": [
#            "What exercises are safe during pregnancy?",
#            "How can I lose weight after childbirth?",
#            "What should I eat to stay healthy?",
#            "Can I exercise while breastfeeding?",
#            "What are some safe exercises to do during pregnancy?",
#            "How soon can I start exercising after a C-section?"
#        ]
#    }
#    return examples
#
def test_model():

    model, scaler, sentence_model = load_models()
    if model is None or scaler is None or sentence_model is None:
        return
    
    # Test örnekleri
    test_queries = [
        "What are the signs of postpartum depression?",
        "Can you help me create a meal plan?",
        "I'm feeling really overwhelmed",
        "What exercises can I do during pregnancy?",
        "How do I know if my baby is healthy?",
        "I need help with my diet",
        "I'm feeling sad and lonely",
        "What should I eat to lose weight?",
        "I am 6 months pregnant and feel pain in my vagina is it normal?"
    ]
    

    
    for query in test_queries:
        intent, confidence = predict_intent(query, sentence_model, scaler, model)
        print(f"Soru: {query}")
        print(f"Niyet: {intent} (Güven: {confidence:.3f})")
        print("-" * 40)

def interactive_test():
    
    # Modelleri yükle
    model, scaler, sentence_model = load_models()
    if model is None or scaler is None or sentence_model is None:
        print("Modeller yüklenemedi. Önce eğitim yapın.")
        return
    
    print("\nKategori açıklamaları:")
    print("- health_rag_info: Sağlık bilgileri ve tıbbi sorular")
    print("- nutrition: Beslenme planları ve diyet önerileri")
    print("- emotional_support: Duygusal destek ve psikolojik yardım")
    print("- diet_exercise: Egzersiz ve diyet kombinasyonu")
    print("- anlasilamadi: Anlaşılamayan mesajlar")
    


    
    print("\n" + "="*60)
    print("Mesajınızı yazın (çıkmak için 'quit' yazın):")
    print("="*60)
    
    while True:
        try:
            user_input = input("\nMesajınız: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'çık', 'çıkış']:
                print("Test sonlandırılıyor...")
                break
            
            if not user_input:
                print("Lütfen bir mesaj yazın.")
                continue
            
            # Niyet tahmini yap
            intent, confidence = predict_intent(user_input, sentence_model, scaler, model)
            
            print(f"\n🔍 TAHMİN SONUCU:")
            print(f"📝 Mesaj: {user_input}")
            print(f"🎯 Kategori: {intent}")
            print(f"📊 Güven Skoru: {confidence:.3f} ({confidence*100:.1f}%)")
            
          
            
        except KeyboardInterrupt:
            print("\n\nTest sonlandırılıyor...")
            break
        except Exception as e:
            print(f"Hata oluştu: {e}")
            continue

def compare_methods():

    test_messages = [
        "I am 6 months pregnant and feel pain in my vagina is it normal?",
        "Can you create a meal plan for breastfeeding?",
        "I'm feeling overwhelmed as a new mom",
        "What exercises are safe during pregnancy?"
    ]
    
    print("Test mesajları:")
    for i, msg in enumerate(test_messages, 1):
        print(f"{i}. {msg}")
    
    print("\n" + "="*80)
    print("SONUÇLAR:")
    print("="*80)
    
    # Sentence Transformer sonuçları
    model, scaler, sentence_model = load_models()
    if model and scaler and sentence_model:
        print("\n🔬 SENTENCE TRANSFORMER SONUÇLARI:")
        for msg in test_messages:
            intent, confidence = predict_intent(msg, sentence_model, scaler, model)
            print(f"📝 {msg}")
            print(f"🎯 {intent} ({confidence:.3f})")
            print("-" * 40)

if __name__ == "__main__":
    
    while True:
        print("\n" + "="*60)
        print("MENÜ:")
        print("1. Model eğitimi yap")
        print("2. Otomatik test çalıştır")
        print("3. İnteraktif test")
        print("4. Yöntem karşılaştırması")
        print("5. Çıkış")
        print("="*60)
        
        choice = input("\nSeçiminizi yapın (1-5): ").strip()
        
        if choice == "1":
            train_and_save_model()
            print("\nModel eğitimi tamamlandı!")
            
        elif choice == "2":
            test_model()
            
        elif choice == "3":
            interactive_test()
            
        elif choice == "4":
            compare_methods()
            
        elif choice == "5":
            print("Uygulama kapatılıyor...")
            break
            
        else:
            print("Geçersiz seçim! Lütfen 1-5 arası bir sayı girin.") 