# Proje Analiz Raporu (LaTeX / IEEE)

Bu klasör, projenin akademik analiz raporunu IEEE konferans şablonu (`IEEEtran`) ile içerir.

## İçerik

```
report/
├── main.tex          # Ana LaTeX dosyası (IEEE konferans şablonu)
├── references.bib    # BibTeX referansları (literatür)
├── figures/          # Eğitim eğrileri ve confusion matrix
│   ├── brain_2d_dice.png
│   ├── brain_2d_loss.png
│   ├── brain_3d_dice.png
│   ├── brain_3d_loss.png
│   ├── lung_accuracy.png
│   ├── lung_loss.png
│   └── lung_confusion.png
└── README.md         # Bu dosya
```

## Overleaf'te derleme

1. https://www.overleaf.com adresine git, oturum aç
2. **New Project → Upload Project**
3. Bu `report/` klasörünün içindeki dosyaları **bir ZIP olarak** sıkıştır ve yükle
   - Tüm dosyalar dahil: `main.tex`, `references.bib`, `figures/*`
4. Overleaf otomatik olarak `main.tex`'i ana dosya olarak tanıyacak
5. **Compiler**: `pdfLaTeX` (varsayılan, sorunsuz)
6. **TeX Live version**: 2023 veya üzeri tavsiye edilir

İlk compile sırasında **iki kez** derlemek gerekebilir (bibliyografya için):
- LaTeX → BibTeX → LaTeX → LaTeX (Overleaf "Recompile from scratch" otomatik yapar)

## Yerelde derleme

`texlive-full` veya `MiKTeX` kurulu bir Windows/Linux makinede:

```bash
cd report
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Çıktı: `main.pdf`

## Önemli not — neden `babel/turkish` kullanılmadı?

`\usepackage[turkish]{babel}` paketi bazı LaTeX kurulumlarında `\includegraphics`
ile birlikte kullanıldığında **figürlerin width/scale parametrelerini görmezden gelir**
veya hatalı uygular. Bu durum özellikle Overleaf'in farklı TeX Live sürümlerinde
tutarsız davranıyor.

Bunun yerine Türkçe karakter desteği için şu kombinasyon kullanılıyor:

```latex
\usepackage[utf8]{inputenc}    % UTF-8 girdi
\usepackage[T1]{fontenc}       % T1 font encoding (ç, ğ, ş, ı, İ, ö, ü desteği)
\usepackage{lmodern}           % Yumuşak vektör fontu
```

Bu kombinasyon Türkçe karakterleri sorunsuz işler; tireleme (hyphenation) Türkçe
kurallarına göre yapılmaz ama bu rapor için anlamlı bir problem değil.

## PDF'i paylaşırken

Overleaf'te derledikten sonra üst menüden **Download PDF** ile `main.pdf` indirilir.
Bu PDF'i e-postaya ek olarak gönderebilir veya repoya `report/main.pdf` olarak
ekleyebilirsin (boyut küçük olduğu için commitlemek sorun değil).
