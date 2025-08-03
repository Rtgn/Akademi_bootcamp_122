# EmpaMom - Anne ve Bebek Takip Uygulaması

EmpaMom, hamilelik ve doğum sonrası dönemde annelere destek olmak için tasarlanmış kapsamlı bir mobil uygulamadır. AI destekli chatbot, su takibi, beslenme planları ve daha fazlasını içerir.

## 🚀 Özellikler

### 📱 Ana Özellikler
- **AI Chatbot**: EmpaMom asistanı ile sorularınızı sorun
- **Su Takibi**: Günlük su tüketiminizi takip edin
- **Beslenme Planları**: Kişiselleştirilmiş beslenme önerileri
- **Bebek Takibi**: Bebek gelişim aşamalarını izleyin
- **Anne Sağlığı**: Doğum sonrası sağlık takibi

### 🤖 Chatbot Modülleri
- **Beslenme Modülü**: Kişiselleştirilmiş beslenme planları
- **Sağlık Bilgisi Modülü**: Hamilelik ve doğum sonrası bilgiler
- **Diyet ve Egzersiz Modülü**: Sağlıklı yaşam önerileri
- **Duygusal Destek Modülü**: Psikolojik destek ve motivasyon

## 🛠️ Kurulum

### Gereksinimler
- Flutter SDK (3.8.1+)
- Python 3.8+
- CUDA destekli GPU (opsiyonel, CPU da çalışır)

### 1. Flutter Uygulaması

```bash
# Projeyi klonlayın
git clone <repository-url>
cd momempa

# Bağımlılıkları yükleyin
flutter pub get

# Uygulamayı çalıştırın
flutter run
```

### 2. Chatbot Backend

```bash
# Backend klasörüne gidin
cd chatbot_backend

# Python bağımlılıklarını yükleyin
pip install -r requirements_api.txt

# API sunucusunu başlatın
python api_chatbot.py
```

Backend sunucusu `http://localhost:5000` adresinde çalışacaktır.

## 📁 Proje Yapısı

```
momempa/
├── lib/                    # Flutter uygulama kodu
│   ├── constans/          # Renk ve sabitler
│   ├── model/             # Veri modelleri
│   ├── view/              # UI bileşenleri
│   │   ├── screens/       # Sayfa ekranları
│   │   └── widget/        # Yeniden kullanılabilir widget'lar
│   └── viewmodel/         # State management
├── chatbot_backend/        # Python chatbot backend
│   ├── api_chatbot.py     # Flask API sunucusu
│   ├── main_chatbot.py    # Terminal chatbot
│   ├── sentence_transformer_intent.py  # Niyet tanıma
│   └── modules/           # Chatbot modülleri
│       ├── nutrition/     # Beslenme modülü
│       ├── health_rag_info/  # Sağlık bilgisi
│       ├── diet_exercise/    # Diyet ve egzersiz
│       └── emotional_support/ # Duygusal destek
└── assets/                # Uygulama varlıkları
    └── lottie/           # Animasyonlar
```

## 🔧 API Endpoint'leri

### Chat API
- **POST** `/api/chat` - Ana chat endpoint'i
- **POST** `/api/nutrition/answer` - Beslenme sorularına cevap
- **GET** `/api/health` - Sağlık kontrolü
- **GET** `/api/modules` - Modül listesi

### Örnek Kullanım
```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Bebeğim çok ağlıyor, ne yapmalıyım?"}'
```

## 🎨 Tasarım

Uygulama modern ve kullanıcı dostu bir tasarıma sahiptir:
- **Renk Paleti**: Mor tonları (EmpaMom teması)
- **Tipografi**: Poppins font ailesi
- **Animasyonlar**: Lottie animasyonları
- **Responsive**: Tüm ekran boyutlarına uyumlu

## 🤝 Katkıda Bulunma

1. Projeyi fork edin
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add amazing feature'`)
4. Branch'inizi push edin (`git push origin feature/amazing-feature`)
5. Pull Request oluşturun

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## 📞 İletişim

- **Proje Sahibi**: [Adınız]
- **Email**: [email@example.com]
- **GitHub**: [github-username]

## 🙏 Teşekkürler

- Flutter ekibine
- Python topluluğuna
- Tüm katkıda bulunanlara

---

**EmpaMom ile annelik yolculuğunuzda yanınızdayız! 💜**
