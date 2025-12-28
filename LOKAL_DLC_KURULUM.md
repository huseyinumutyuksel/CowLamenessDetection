# 🐄 Lokal DeepLabCut Kurulum ve Çalıştırma Rehberi

## Adım 1: Python Ortamı Oluştur

### Windows:
```cmd
# Anaconda/Miniconda kurulu olmalı
conda create -n dlc python=3.10
conda activate dlc
```

### Mac/Linux:
```bash
conda create -n dlc python=3.10
conda activate dlc
```

## Adım 2: DeepLabCut Kur

```bash
pip install deeplabcut
```

**Not**: Kurulum ~5-10 dakika sürebilir.

## Adım 3: Google Drive Desktop Kur (Opsiyonel ama Önerilen)

1. [Google Drive Desktop](https://www.google.com/drive/download/) indir
2. Kur ve Google hesabınla giriş yap
3. "Bilgisayarımla senkronize et" seçeneğini seç
4. `My Drive` klasörü lokal bilgisayarınızda görünecek

**Windows örnek yol:**
```
C:\Users\YourName\Google Drive\My Drive\Inek Topallik Tespiti Parcalanmis Inek Videolari\cow_single_videos\
```

**Mac örnek yol:**
```
/Users/YourName/Google Drive/My Drive/Inek Topallik Tespiti Parcalanmis Inek Videolari/cow_single_videos/
```

## Adım 4: Script'i Düzenle

1. `run_dlc_local.py` dosyasını aç
2. Satır 26'yı güncelle:
   ```python
   BASE_VIDEO_DIR = "C:/Users/YourName/Google Drive/My Drive/.../cow_single_videos"
   ```

## Adım 5: Script'i Çalıştır

```bash
# Ortamı aktifleştir
conda activate dlc

# Script'i çalıştır
python run_dlc_local.py
```

**Ne Olacak?**
1. Script 1168 videoyu bulacak
2. Süre tahmini verecek (~39 saat)
3. Her video için CSV dosyası oluşturacak
4. İlerleme çubuğu gösterecek

## Adım 6: İşlemi İzle

### Resume (Devam Etme) Özelliği
Bilgisayar kapanırsa veya işlem kesilirse:
```bash
python run_dlc_local.py
```
Script otomatik olarak işlenmiş videoları atlayıp kaldığı yerden devam eder.

### İlerleme Kontrolü
CSV dosyalarının oluşup oluşmadığını kontrol edin:
```bash
# Windows
dir "cow_single_videos\Saglikli\*DLC*.csv"

# Mac/Linux  
ls cow_single_videos/Saglikli/*DLC*.csv | wc -l
```

## Adım 7: CSV'leri Drive'a Yükle

### Seçenek A: Google Drive Desktop Kullanıyorsanız
✅ **Hiçbir şey yapmayın!** CSV'ler otomatik olarak senkronize olacak.

### Seçenek B: Manuel Yükleme
1. Google Drive web arayüzünü aç
2. `cow_single_videos/Saglikli/` ve `cow_single_videos/Topal/` klasörlerine git
3. `*DLC*.csv` dosyalarını yükle

## Performans İpuçları

### Bilgisayar Özellikleri
- **Minimum**: 8GB RAM, 4 core CPU
- **Önerilen**: 16GB+ RAM, 8+ core CPU
- **GPU**: Gerekli değil (SuperAnimal inference CPU'da hızlı)

### İşlem Süresi (1168 video)
- **Laptop (4 core, 8GB RAM)**: ~78 saat (3.25 gün)
- **Desktop (8 core, 16GB RAM)**: ~39 saat (1.6 gün)
- **Workstation (16 core, 32GB RAM)**: ~20 saat

### Arka Planda Çalıştırma

**Windows:**
```cmd
# PowerShell
Start-Process python run_dlc_local.py -WindowStyle Hidden
```

**Mac/Linux:**
```bash
nohup python run_dlc_local.py > dlc_log.txt 2>&1 &
```

## Sorun Giderme

### Hata: "No module named 'deeplabcut'"
```bash
conda activate dlc
pip install deeplabcut
```

### Hata: "Video directory not found"
Script'teki `BASE_VIDEO_DIR` yolunu kontrol edin.

### İşlem Çok Yavaş
- Arka plan uygulamalarını kapatın
- Virüs tarama programını duraklatın
- Bilgisayarı prize takın (laptop)

### CSV Dosyaları Oluşmuyor
Videolardan birinde sorun olabilir. Script loguna bakın:
```bash
python run_dlc_local.py > log.txt 2>&1
```

## Sonraki Adım: Colab

CSV'ler Drive'da olduktan sonra:
1. Colab'de `01_Cow_Lameness_Training_v16.ipynb` aç
2. Cell'leri sırayla çalıştır
3. DLC kurulum hücresi CSV'leri bulacak ve "✅ Found XYZ CSV files" mesajı verecek
4. DLC analysis phase otomatik olarak atlanacak
5. Training direkt başlayacak

**Hazırsınız!** 🎉
