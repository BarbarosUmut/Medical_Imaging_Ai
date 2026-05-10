# Medical Imaging AI

Beyin MR tümör segmentasyonu ve Akciğer X-Ray patoloji sınıflandırması yapan, Ollama LLM ile **Türkçe radyoloji raporu** üreten Gradio tabanlı web uygulaması.

## Özellikler

| Modül | Görev | Mimari |
|---|---|---|
| Beyin MR Segmentasyon | Tümör segmentasyonu | 2D U-Net + 3D U-Net (MONAI) |
| Akciğer X-Ray Sınıflandırma | 14 patoloji tespiti | DenseNet-121 (CheXNet) |
| LLM Rapor Üretici | Türkçe klinik rapor | Ollama (yerel) |
| Web Arayüzü | Kullanıcı arayüzü | Gradio |

## Gereksinimler

- **Python 3.12**
- **Ollama** (yerel sunucu — `http://localhost:11434`)
- **Platform**: Windows 11 (Linux/macOS'ta da çalışmalı)

## Kurulum

```bash
# Repoyu klonla
git clone https://github.com/BarbarosUmut/Medical_Imaging_Ai.git
cd Medical_Imaging_Ai

# Sanal ortam oluştur
python -m venv venv

# Aktive et (Windows PowerShell)
.\venv\Scripts\Activate.ps1
# (Linux/macOS)
# source venv/bin/activate

# Bağımlılıkları yükle
pip install -r requirements.txt

# Ollama modelini çek
ollama pull llama3
```

## Çalıştırma

```bash
python app/main.py
```

Tarayıcıdan **http://localhost:7860** adresini aç.

## Klasör Yapısı

```
medical_imaging_ai/
├── app/                # Gradio web uygulaması (giriş noktası: app/main.py)
├── inference/          # Çıkarım modülleri (BrainSegmentor, LungClassifier)
├── llm/                # Ollama tabanlı rapor üretici
├── models/             # Eğitilmiş model ağırlıkları (.pth)
├── training/           # Eğitim scriptleri
├── utils/              # Görüntü ön işleme, metrikler, görselleştirme
├── tests/              # pytest birim testleri
└── requirements.txt
```

> `data/` klasörü büyük veri setleri içerdiği için repoya dahil edilmemiştir. BraTS (beyin MR) ve NIH ChestX-ray14 (akciğer) veri setlerini kendi makinende `data/` altına yerleştirebilirsin.

## Modeller

Önceden eğitilmiş ağırlıklar `models/` klasörü altında repo ile birlikte gelir:

- `models/brain/attnunet2d_best.pth` — Beyin MR 2D Attention U-Net
- `models/brain/attnunet3d_best.pth` — Beyin MR 3D Attention U-Net
- `models/lung/chexnet_best.pth` — Akciğer DenseNet-121 (CheXNet)

## Testler

```bash
# Tüm testleri çalıştır
python -m pytest tests/ -v

# Tek modül
python -m pytest tests/test_brain.py -v
```

## Mimari Notlar

- **MONAI transforms**: eğitimde `RandFlipd`, `RandRotate90d`, `NormalizeIntensityd`; çıkarımda `Spacingd` + `NormalizeIntensityd`.
- **GPU/CPU**: `torch.device("cuda" if torch.cuda.is_available() else "cpu")` her modülde kontrol edilir.
- **Türkçe raporlar**: LLM prompt'unda "Türkçe radyoloji raporu yaz" direktifi kullanılır.
- **Sliding window inference**: 3D segmentasyon için `monai.inferers.SlidingWindowInferer`.

## Lisans

[MIT](LICENSE)

## ⚠️ Sorumluluk Reddi

Bu yazılım **yalnızca araştırma ve eğitim amaçlıdır**. Klinik tanı veya tedavi kararları için kullanılmamalıdır. Tıbbi görüntüleme yorumlamasını her zaman lisanslı bir radyolog yapmalıdır.
