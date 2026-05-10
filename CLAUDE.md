# Medical Imaging AI — CLAUDE.md

## Proje Özeti

Beyin MR tümör segmentasyonu ve Akciğer X-Ray sınıflandırması yapan, Ollama LLM ile Türkçe radyoloji raporu üreten Gradio tabanlı web uygulaması.

### Bileşenler

| Bileşen | Görev | Mimari |
|---|---|---|
| Beyin MR Segmentasyon | Tümör segmentasyonu | 2D U-Net + 3D U-Net (MONAI) |
| Akciğer X-Ray Sınıflandırma | Patoloji tespiti | DenseNet-121 (CheXNet) |
| LLM Rapor Üretici | Türkçe klinik rapor | Ollama (yerel) |
| Web Arayüzü | Kullanıcı arayüzü | Gradio |

---

## Ortam

- **Python**: 3.12
- **Deep Learning**: PyTorch + MONAI
- **Web**: Gradio
- **LLM**: Ollama (yerel sunucu, `http://localhost:11434`)
- **Platform**: Windows 11

### Ortam Kurulumu

```bash
# Sanal ortam oluştur
python -m venv venv
venv\Scripts\activate

# Bağımlılıkları yükle
pip install -r requirements.txt

# Ollama modelini çek (örn. llama3 veya mistral)
ollama pull llama3
```

### requirements.txt İçermesi Gerekenler

```
torch>=2.2.0
torchvision
monai[all]>=1.3.0
gradio>=4.0
nibabel          # NIfTI dosya formatı (beyin MR)
pydicom          # DICOM dosya formatı
pillow
numpy
scikit-learn
matplotlib
requests         # Ollama API çağrıları için
```

---

## Klasör Yapısı

```
medical_imaging_ai/
│
├── data/                    # Ham ve işlenmiş veriler (git'e ekleme)
│   ├── brain/               # Beyin MR veri seti (NIfTI .nii.gz)
│   │   ├── raw/
│   │   └── processed/
│   └── lung/                # Akciğer X-Ray veri seti (JPEG/PNG)
│       ├── raw/
│       └── processed/
│
├── models/                  # Eğitilmiş model ağırlıkları (.pth)
│   ├── brain/
│   │   ├── unet2d_best.pth
│   │   └── unet3d_best.pth
│   └── lung/
│       └── chexnet_best.pth
│
├── training/                # Eğitim scriptleri
│   ├── train_brain_2d.py
│   ├── train_brain_3d.py
│   └── train_lung.py
│
├── inference/               # Çıkarım (inference) modülleri
│   ├── brain_segmentor.py   # BrainSegmentor sınıfı
│   └── lung_classifier.py   # LungClassifier sınıfı
│
├── llm/                     # LLM rapor üretici
│   └── report_generator.py  # OllamaReportGenerator sınıfı
│
├── app/                     # Gradio web uygulaması
│   ├── main.py              # Giriş noktası — `python app/main.py`
│   └── ui_components.py     # Gradio blok tanımları
│
├── utils/                   # Yardımcı fonksiyonlar
│   ├── preprocessing.py     # Görüntü ön işleme
│   ├── visualization.py     # Segmentasyon maskesi görselleştirme
│   └── dicom_utils.py       # DICOM yardımcıları
│
├── tests/                   # Birim ve entegrasyon testleri
│   ├── test_brain.py
│   ├── test_lung.py
│   └── test_report.py
│
├── requirements.txt
├── CLAUDE.md
└── .gitignore               # data/ ve models/ dahil etme
```

---

## Modül Detayları

### inference/brain_segmentor.py — `BrainSegmentor`

- MONAI `UNet` ile 2D ve 3D segmentasyon.
- Giriş: NIfTI (`.nii.gz`) veya DICOM dizini.
- Çıkış: ikili maske array + Dice skoru.
- Sliding window inference (`monai.inferers.SlidingWindowInferer`) kullanılacak.
- 3D modelde kanal sayısı: `(1, 64, 128, 256, 256)` encoder; decoder simetrik.

### inference/lung_classifier.py — `LungClassifier`

- `torchvision.models.densenet121` base, son katman 14 sınıf (CheXNet protokolü).
- Giriş: PIL Image veya dosya yolu.
- Çıkış: 14 patoloji için olasılık sözlüğü + en yüksek 3 bulgu.
- ImageNet normalizasyonu: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`.
- Görüntüler 224×224 yeniden boyutlandırılacak.

### llm/report_generator.py — `OllamaReportGenerator`

- Ollama REST API: `POST http://localhost:11434/api/generate`.
- Model: yapılandırılabilir (varsayılan `llama3`).
- Çıkarım sonuçlarını (segmentasyon + sınıflandırma bulgularını) birleştirip **Türkçe** klinik radyoloji raporu üretir.
- Prompt şablonu `llm/` içinde `.txt` veya `.jinja2` dosyası olarak tutulacak.
- Streaming response desteklenecek.

### app/main.py

- Uygulamayı başlatmak için: `python app/main.py`
- Gradio `Blocks` API kullanılacak (Tabs: Beyin MR | Akciğer X-Ray | Rapor).
- Model yükleme uygulama başlangıcında bir kez yapılacak; her istek için yeniden yüklenmeyecek.

---

## Kritik Tasarım Kararları

- **MONAI transforms**: eğitimde `RandFlipd`, `RandRotate90d`, `NormalizeIntensityd` kullanılacak; çıkarımda yalnızca `Spacingd` + `NormalizeIntensityd`.
- **Veri seti yolları**: `data/` altında sabit; ortam değişkeni veya config dosyası yok — yollar `utils/preprocessing.py` içinde sabit tanımlanacak.
- **GPU/CPU**: `torch.device("cuda" if torch.cuda.is_available() else "cpu")` her modülde ayrı kontrol edilecek.
- **Model kaydetme**: yalnızca `model.state_dict()` kaydedilecek; tam model nesnesi değil.
- **Türkçe raporlar**: LLM çıktısı Türkçe olacak; prompt'ta "Türkçe radyoloji raporu yaz" direktifi bulunacak.

---

## Testler

```bash
# Tüm testleri çalıştır
python -m pytest tests/ -v

# Tek modül
python -m pytest tests/test_brain.py -v
```

- `tests/` içindeki testler gerçek model ağırlıklarına ihtiyaç duymaz; küçük sentetik tensor'larla çalışır.
- Ollama testleri `unittest.mock` ile mocklenir.

---

## .gitignore Notları

`data/` ve `models/` klasörleri git'e eklenmez (büyük dosyalar). Model ağırlıkları ayrı bir artifact deposunda tutulacak.

---

## Sık Kullanılan Komutlar

```powershell
# Uygulamayı başlat
python app/main.py

# Beyin modeli eğit
python training/train_brain_2d.py

# Akciğer modeli eğit
python training/train_lung.py

# Ollama sunucusu çalışıyor mu kontrol et
curl http://localhost:11434/api/tags
```
