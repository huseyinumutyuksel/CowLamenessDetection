# 🐄 Cow Lameness Detection & Segmentation (v16)
### Academic Gold Standard System | Akademik Altın Standart Sistem

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![DeepLabCut](https://img.shields.io/badge/DeepLabCut-SuperAnimal-green.svg)](https://www.mackenziemathislab.org/deeplabcut)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[English](#english) | [Türkçe](#turkish)

---

<a name="english"></a>
## 🇬🇧 English

### Overview
State-of-the-art deep learning system for automated cow lameness detection and segmentation using **Tri-Modal Gait Analysis**. Designed for academic research and production deployment on Google Colab.

### 🎯 Key Features

**Academic Validation**:
- ✅ DeepLabCut SuperAnimal-Quadruped for pose estimation
- ✅ 5-Fold Cross-Validation with train/test split (80/20)
- ✅ Biometric Statistical Analysis (T-Test on back arch angle)
- ✅ Ablation Study (Pose-Only vs VideoMAE-Only vs Tri-Modal)
- ✅ ROC-AUC curves and comprehensive metrics
- ✅ t-SNE visualization for feature space analysis
- ✅ Publication-ready results with unbiased test set evaluation

**Production Features**:
- 🎬 Multi-cow tracking with ByteTrack
- 🎨 High-quality segmentation with SAM (Segment Anything)
- 📊 Clinical CSV reports per cow
- 🎮 GPU-accelerated inference
- 📈 Real-time FPS monitoring
- 💾 Automatic Drive caching for DLC results

### 🏗️ Architecture

**Tri-Modal Feature Fusion**:
1. **Structure (Pose)**: DeepLabCut SuperAnimal → Skeletal keypoints
2. **Deep Motion**: VideoMAE V2 → Spatiotemporal features
3. **Pure Motion**: RAFT Optical Flow → Movement patterns

**Fusion**: Temporal Transformer Encoder → Binary Classification (Healthy/Lame)

### 📦 Installation

**Google Colab (Recommended)**:
```python
!pip install ultralytics timm einops transformers
!pip install "deeplabcut[tf]"
!pip install segment-anything supervision
!pip install moviepy scikit-learn scipy seaborn matplotlib psutil gputil
```

**SAM Checkpoint**:
```bash
!wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

### 📁 Dataset Structure

```
Google Drive/
├── Inek Topallik Tespiti Parcalanmis Inek Videolari/
│   └── cow_single_videos/
│       ├── Saglikli/          # Healthy cows
│       │   ├── video1.mp4
│       │   └── ...
│       └── Topal/             # Lame cows
│           ├── video1.mp4
│           └── ...
├── Raw_MultiCow_Videos/       # Multi-cow videos for inference
│   └── test_video.mp4
└── outputs_v16_academic/      # Results (auto-created)
```

### 🚀 Usage

#### 1. Training Notebook (`01_Cow_Lameness_Training_v16.ipynb`)

**Steps**:
1. Upload to Google Colab with **GPU Runtime**
2. Mount Google Drive
3. Run all cells (estimated time: 3-5 hours for 100 videos)

**Outputs**:
- ✅ Trained model: `cow_gait_transformer_v16_final.pth`
- 📊 Biometric plot: `biometric_significance.png`
- 📈 ROC-AUC curve: `roc_auc_curve.png`
- 🎨 t-SNE clusters: `tsne_clusters.png`
- 📉 Loss curves: `loss_curves.png`
- 🔬 Ablation study: `ablation_study.png`
- 🧮 Confusion matrix: `confusion_matrix.png`

**Key Metrics**:
- 5-Fold CV Accuracy (mean ± std)
- **Test Set Accuracy** (unbiased, publication-ready)
- Statistical significance (p-value < 0.05)

#### 2. Inference Notebook (`02_Cow_Lameness_Inference_Multi_v16.ipynb`)

**Steps**:
1. Ensure trained model exists from Notebook 1
2. Upload multi-cow video to `Raw_MultiCow_Videos/`
3. Run all cells

**Outputs**:
- 🎬 Annotated video: `inference_result_v16.mp4`
  - Red masks = Lame cows
  - Green masks = Healthy cows
  - Unique IDs for tracking
- 📄 Clinical report: `clinical_report_v16.csv`

**Report Format**:
```csv
Cow_ID,Diagnosis,Confidence,Frames_Tracked,Duration_Seconds
1,SAGLIKLI (HEALTHY),0.8523,450,15.0
2,TOPAL (LAME),0.9102,380,12.67
```

### 🧪 Academic Validation

#### Biometric Analysis
- **Metric**: Hip-Spine-Shoulder angle
- **Test**: Independent samples T-Test
- **Null Hypothesis**: No difference between healthy/lame groups
- **Result**: p-value with KDE visualization

#### Cross-Validation
- **Method**: Stratified 5-Fold CV on 80% training set
- **Final Test**: 20% held-out set (never seen during training)
- **Reported**: Mean ± Std Dev across folds + final test accuracy

#### Ablation Study
Comparison of model variants:
- Pose-Only Model
- VideoMAE-Only Model
- **Tri-Modal (Ours)** ← Best performance

### 📊 System Requirements

**Minimum** (Google Colab Free):
- GPU: Tesla T4 (16GB VRAM)
- RAM: 12GB

**Recommended** (Colab Pro+):
- GPU: A100 (40-80GB VRAM)
- RAM: 150GB+
- Faster processing for large datasets

### 🔧 Troubleshooting

**Issue**: DLC dimension mismatch  
**Solution**: Ensure both notebooks use same DLC model (SuperAnimal-Quadruped)

**Issue**: Out of memory  
**Solution**: Reduce batch size or use Colab Pro+

**Issue**: Session timeout during DLC  
**Solution**: Results cached in Drive, re-run skips processed videos

### 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{cow_lameness_v16,
  title={Cow Lameness Detection using Tri-Modal Gait Analysis},
  author={Your Name},
  year={2025},
  version={v16},
  url={https://github.com/yourusername/cow-lameness-detection}
}
```

### 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch
3. Submit pull request with detailed description

### 📄 License

MIT License - see [LICENSE](LICENSE) file

### 🙏 Acknowledgments

- DeepLabCut team for SuperAnimal model
- Meta AI for Segment Anything Model (SAM)
- Hugging Face for VideoMAE implementation

---

<a name="turkish"></a>
## 🇹🇷 Türkçe

### Genel Bakış
**Tri-Modal Yürüyüş Analizi** kullanarak otomatik inek topallık tespiti ve segmentasyonu için son teknoloji derin öğrenme sistemi. Akademik araştırma ve Google Colab'de production deployment için tasarlanmıştır.

### 🎯 Temel Özellikler

**Akademik Doğrulama**:
- ✅ Pose tahmini için DeepLabCut SuperAnimal-Quadruped
- ✅ Train/test ayrımı ile 5-Katlı Çapraz Doğrulama (%80/%20)
- ✅ Biyometrik İstatistiksel Analiz (sırt kavisi açısı üzerinde T-Testi)
- ✅ Ablation Study (Sadece-Pose vs Sadece-VideoMAE vs Tri-Modal)
- ✅ ROC-AUC eğrileri ve kapsamlı metrikler
- ✅ Özellik uzayı analizi için t-SNE görselleştirme
- ✅ Tarafsız test seti değerlendirmesi ile yayına hazır sonuçlar

**Production Özellikleri**:
- 🎬 ByteTrack ile çoklu inek takibi
- 🎨 SAM (Segment Anything) ile yüksek kaliteli segmentasyon
- 📊 İnek bazında klinik CSV raporları
- 🎮 GPU-hızlandırılmış inference
- 📈 Gerçek zamanlı FPS izleme
- 💾 DLC sonuçları için otomatik Drive önbellekleme

### 🏗️ Mimari

**Tri-Modal Özellik Füzyonu**:
1. **Yapı (Pose)**: DeepLabCut SuperAnimal → İskelet anahtar noktaları
2. **Derin Hareket**: VideoMAE V2 → Uzay-zamansal özellikler
3. **Saf Hareket**: RAFT Optik Akış → Hareket desenleri

**Füzyon**: Temporal Transformer Encoder → İkili Sınıflandırma (Sağlıklı/Topal)

### 📦 Kurulum

**Google Colab (Önerilen)**:
```python
!pip install ultralytics timm einops transformers
!pip install "deeplabcut[tf]"
!pip install segment-anything supervision
!pip install moviepy scikit-learn scipy seaborn matplotlib psutil gputil
```

**SAM Checkpoint**:
```bash
!wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

### 📁 Veri Seti Yapısı

```
Google Drive/
├── Inek Topallik Tespiti Parcalanmis Inek Videolari/
│   └── cow_single_videos/
│       ├── Saglikli/          # Sağlıklı inekler
│       │   ├── video1.mp4
│       │   └── ...
│       └── Topal/             # Topal inekler
│           ├── video1.mp4
│           └── ...
├── Raw_MultiCow_Videos/       # Inference için çoklu inek videoları
│   └── test_video.mp4
└── outputs_v16_academic/      # Sonuçlar (otomatik oluşturulur)
```

### 🚀 Kullanım

#### 1. Eğitim Notebook'u (`01_Cow_Lameness_Training_v16.ipynb`)

**Adımlar**:
1. **GPU Runtime** ile Google Colab'e yükleyin
2. Google Drive'ı bağlayın
3. Tüm hücreleri çalıştırın (tahmini süre: 100 video için 3-5 saat)

**Çıktılar**:
- ✅ Eğitilmiş model: `cow_gait_transformer_v16_final.pth`
- 📊 Biyometrik grafik: `biometric_significance.png`
- 📈 ROC-AUC eğrisi: `roc_auc_curve.png`
- 🎨 t-SNE kümeleri: `tsne_clusters.png`
- 📉 Kayıp eğrileri: `loss_curves.png`
- 🔬 Ablation çalışması: `ablation_study.png`
- 🧮 Karmaşıklık matrisi: `confusion_matrix.png`

**Temel Metrikler**:
- 5-Katlı CV Doğruluğu (ortalama ± standart sapma)
- **Test Seti Doğruluğu** (tarafsız, yayına hazır)
- İstatistiksel anlamlılık (p-değeri < 0.05)

#### 2. Inference Notebook'u (`02_Cow_Lameness_Inference_Multi_v16.ipynb`)

**Adımlar**:
1. Notebook 1'den eğitilmiş modelin mevcut olduğundan emin olun
2. Çoklu inek videosunu `Raw_MultiCow_Videos/` klasörüne yükleyin
3. Tüm hücreleri çalıştırın

**Çıktılar**:
- 🎬 Açıklamalı video: `inference_result_v16.mp4`
  - Kırmızı maskeler = Topal inekler
  - Yeşil maskeler = Sağlıklı inekler
  - Takip için benzersiz ID'ler
- 📄 Klinik rapor: `clinical_report_v16.csv`

**Rapor Formatı**:
```csv
Cow_ID,Diagnosis,Confidence,Frames_Tracked,Duration_Seconds
1,SAGLIKLI (HEALTHY),0.8523,450,15.0
2,TOPAL (LAME),0.9102,380,12.67
```

### 🧪 Akademik Doğrulama

#### Biyometrik Analiz
- **Metrik**: Kalça-Omurga-Omuz açısı
- **Test**: Bağımsız örneklemler T-Testi
- **Null Hipotezi**: Sağlıklı/topal gruplar arasında fark yok
- **Sonuç**: KDE görselleştirmesi ile p-değeri

#### Çapraz Doğrulama
- **Yöntem**: %80 eğitim seti üzerinde Stratified 5-Katlı CV
- **Final Test**: %20 ayrılmış set (eğitim sırasında hiç görülmemiş)
- **Raporlanan**: Katlar arası ortalama ± Std Sapma + final test doğruluğu

#### Ablation Çalışması
Model varyantlarının karşılaştırması:
- Sadece-Pose Modeli
- Sadece-VideoMAE Modeli
- **Tri-Modal (Bizimki)** ← En iyi performans

### 📊 Sistem Gereksinimleri

**Minimum** (Google Colab Free):
- GPU: Tesla T4 (16GB VRAM)
- RAM: 12GB

**Önerilen** (Colab Pro+):
- GPU: A100 (40-80GB VRAM)
- RAM: 150GB+
- Büyük veri setleri için daha hızlı işleme

### 🔧 Sorun Giderme

**Sorun**: DLC boyut uyuşmazlığı  
**Çözüm**: Her iki notebook'un da aynı DLC modelini kullandığından emin olun (SuperAnimal-Quadruped)

**Sorun**: Bellek yetersiz  
**Çözüm**: Batch size'ı azaltın veya Colab Pro+ kullanın

**Sorun**: DLC sırasında oturum zaman aşımı  
**Çözüm**: Sonuçlar Drive'da önbellekleniyor, yeniden çalıştırma işlenmiş videoları atlar

### 📚 Alıntı

Bu kodu araştırmanızda kullanırsanız, lütfen alıntı yapın:

```bibtex
@software{cow_lameness_v16,
  title={Tri-Modal Yürüyüş Analizi Kullanarak İnek Topallık Tespiti},
  author={İsminiz},
  year={2025},
  version={v16},
  url={https://github.com/kullaniciadi/cow-lameness-detection}
}
```

### 🤝 Katkıda Bulunma

Katkılar memnuniyetle karşılanır! Lütfen:
1. Repository'yi fork edin
2. Feature branch oluşturun
3. Detaylı açıklama ile pull request gönderin

### 📄 Lisans

MIT Lisansı - [LICENSE](LICENSE) dosyasına bakın

### 🙏 Teşekkürler

- SuperAnimal modeli için DeepLabCut ekibi
- Segment Anything Model (SAM) için Meta AI
- VideoMAE implementasyonu için Hugging Face

---

## 📞 Contact | İletişim

For questions or collaborations:  
Sorular veya işbirlikleri için:

📧 Email: your.email@example.com  
🐙 GitHub: [@yourusername](https://github.com/yourusername)

---

**Made with ❤️ for precision livestock farming**  
**Hassas hayvancılık için ❤️ ile yapıldı**
