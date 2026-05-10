# Medical Imaging AI

Beyin MR tümör segmentasyonu ve Akciğer X-Ray patoloji sınıflandırması yapan, Ollama LLM ile **Türkçe radyoloji raporu** üreten Gradio tabanlı web uygulaması.

## Özellikler

| Modül | Görev | Mimari |
|---|---|---|
| Beyin MR Segmentasyon | Tümör segmentasyonu | 2D U-Net + 3D U-Net (MONAI) |
| Akciğer X-Ray Sınıflandırma | 4 sınıf (COVID / Lung Opacity / Normal / Viral Pneumonia) | DenseNet-121 |
| LLM Rapor Üretici | Türkçe klinik rapor | Ollama  |
| Web Arayüzü | Kullanıcı arayüzü | Gradio |

## Gereksinimler

- **Python 3.12**
- **Deep Learning**: PyTorch + MONAI
- **Web**: Gradio
- **Ollama** (yerel sunucu — `http://localhost:11434`)
- **Platform**: Windows 11 

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

## Ollama Kurulumu 
AI Radyolog asistanının çalışabilmesi için sisteminizde [Ollama](https://ollama.com/)'nın kurulu olması gerekmektedir.

Ollama'yı başlatın ve Llama 3 modelini indirin
```bash
ollama run llama3
```

## requirements.txt İçermesi Gerekenler

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
├── app/                     # Gradio web uygulaması
│   ├── main.py              # Giriş noktası — `python app/main.py`
│   └── ui_components.py     # Gradio blok tanımları
│
├── inference/               # Çıkarım (inference) modülleri
│   ├── brain_segmentor.py   # BrainSegmentor sınıfı
│   └── lung_classifier.py   # LungClassifier sınıfı
|
├── llm/                     # LLM rapor üretici
│   └── report_generator.py  # OllamaReportGenerator sınıfı
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
└── requirements.txt
```
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
- `tests/` içindeki testler gerçek model ağırlıklarına ihtiyaç duymaz; küçük sentetik tensor'larla çalışır.
- Ollama testleri `unittest.mock` ile mocklenir.

## Mimari Notlar

- **MONAI transforms**: eğitimde `RandFlipd`, `RandRotate90d`, `NormalizeIntensityd`; çıkarımda `Spacingd` + `NormalizeIntensityd`.
- **GPU/CPU**: `torch.device("cuda" if torch.cuda.is_available() else "cpu")` her modülde kontrol edilir.
- **Türkçe raporlar**: LLM prompt'unda "Türkçe radyoloji raporu yaz" direktifi kullanılır.
- **Sliding window inference**: 3D segmentasyon için `monai.inferers.SlidingWindowInferer`.

## Lisans

[MIT](LICENSE)

## ⚠️ Sorumluluk Reddi

Bu yazılım **yalnızca araştırma ve eğitim amaçlıdır**. Klinik tanı veya tedavi kararları için kullanılmamalıdır. Tıbbi görüntüleme yorumlamasını her zaman lisanslı bir radyolog yapmalıdır.

## Geliştirici
Umut Barbaros BABAHAN, Yapay Zeka Mühendisliği, Ostim Teknik Üniversitesi.
