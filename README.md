# Medical Imaging AI

Beyin MR tümör segmentasyonu ve Akciğer X-Ray patoloji sınıflandırması yapan, Ollama LLM ile **Türkçe radyoloji raporu** üreten Gradio tabanlı web uygulaması.

## Özellikler

| Modül | Görev | Mimari |
|---|---|---|
| Beyin MR Segmentasyon | Tümör segmentasyonu | 2D U-Net + 3D U-Net (MONAI) |
| Akciğer X-Ray Sınıflandırma | 4 sınıf (COVID / Lung Opacity / Normal / Viral Pneumonia) | DenseNet-121 |
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

## Veri Setleri

Projede iki farklı tıbbi görüntüleme veri seti kullanılmıştır. `data/` klasörü repoya dahil değildir; aşağıdaki kaynaklardan indirip `data/brain/raw/` ve `data/lung/raw/` altına yerleştirebilirsin.

### 🧠 Beyin MR — BraTS 2020

**Brain Tumor Segmentation Challenge 2020** (MICCAI)

- **Konum**: `data/brain/raw/BraTS2020_TrainingData` ve `data/brain/raw/BraTS2020_ValidationData`
- **Format**: NIfTI (`.nii.gz`) — 3D MR hacimleri
- **Modaliteler**: 4 ayrı MR sekansı per vaka
  - **T1** — T1-ağırlıklı
  - **T1ce** — Kontrastlı T1
  - **T2** — T2-ağırlıklı
  - **FLAIR** — Fluid-Attenuated Inversion Recovery
- **Etiketler (segmentasyon maskeleri)**:
  - Necrotic / non-enhancing tumor core
  - Peritumoral edema
  - GD-enhancing tumor
- **Ön işleme**: `training/preprocess_brats.py` ile 3D NIfTI hacimlerinden 2D axial slice'lar çıkarılıp `.npz` formatına dönüştürülür.
- **Resmi kaynak**: [BraTS Challenge](http://braintumorsegmentation.org/)
- **Citation**:
  > B. H. Menze et al., "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)", *IEEE Transactions on Medical Imaging*, 2015.

### 🫁 Akciğer X-Ray — COVID-19 Radiography Database

Qatar University, University of Dhaka ve ortaklarının derlediği göğüs röntgeni veri tabanı.

- **Konum**: `data/lung/raw/COVID-19_Radiography_Dataset/`
- **Format**: PNG, 299×299 piksel
- **4 sınıf ve görüntü dağılımı**:

  | Sınıf | Görüntü Sayısı | Kaynak |
  |---|---:|---|
  | Normal | 10.192 | RSNA + Kaggle |
  | Lung Opacity (non-COVID enfeksiyon) | 6.012 | RSNA |
  | COVID-19 | 3.616 | padchest, SIRM, GitHub, Kaggle, Twitter, vb. |
  | Viral Pneumonia | 1.345 | Kaggle Chest X-Ray (pneumonia) |
  | **Toplam** | **~21.165** | |

- **İlave varlıklar**: Her sınıf için karşılık gelen akciğer maskeleri ve metadata Excel dosyaları (`COVID.metadata.xlsx`, vb.) mevcut.
- **Resmi kaynak**: [Kaggle — COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
- **Citations**:
  > M. E. H. Chowdhury et al., "Can AI help in screening Viral and COVID-19 pneumonia?", *IEEE Access*, vol. 8, pp. 132665–132676, 2020.
  >
  > T. Rahman et al., "Exploring the Effect of Image Enhancement Techniques on COVID-19 Detection using Chest X-ray Images", *arXiv:2012.02238*, 2020.

> **Not**: Kod tabanı (`models/lung/densenet121.py`) 14 sınıflı CheXNet protokolünü de desteklemektedir (`CLASS_NAMES_CHEXNET`); ancak depo ile birlikte gelen `chexnet_best.pth` ağırlığı yukarıdaki **4 sınıflı COVID-19 Radiography** veri setiyle eğitilmiştir.

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
- `models/lung/chexnet_best.pth` — Akciğer DenseNet-121 (4 sınıf: COVID-19 Radiography üzerinde eğitilmiştir)

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
