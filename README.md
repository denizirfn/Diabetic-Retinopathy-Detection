# Diabetic Retinopathy Tespiti Klasik Makine Öğrenmesi Modelleri İle


Retinal fundus görüntülerinden optik disk, kan damarları, mikroanevrizma ve eksüda gibi yapılar **segment edilerek** öznitelikler çıkarılmış; çeşitli makine öğrenmesi modelleri ile **hasta/normal sınıflandırması** yapılmıştır.

---

## 🚀 Özellikler
- **Segmentasyon Adımları:** Optik disk, damar, mikroanevrizma ve eksüda
- **Öznitelik Çıkarımı:**
  - Alan (beyaz piksellerin toplamı)
  - Ortalama piksel değeri
  - Piksel yoğunluğunun standart sapması
  - Doluluk oranı (segmentli bölgenin görüntüye oranı)
  - Kan damar uzunluğu ve kıvrıklığı
  - Kenar yoğunluğu  
- **Makine Öğrenmesi Modelleri:** Random Forest, Support Vector Machines (SVM), Logistic Regression, K-Nearest Neighbors (KNN)

---

## 🧩 Özellik Çıkarımı Detayları
- **Optik Disk Alanı:** Segmentasyon sonucu elde edilen ikili (binary) görüntüde beyaz piksellerin (255 değerindeki piksellerin) sayımı ile hesaplandı.  
- **Mikroanevrizma Doluluk Oranı:** Mikroanevrizmaların kapladığı oransal alan, DR olasılığının güçlü bir göstergesidir.  
- **Kan Damarı Uzunluğu ve Kıvrıklığı:**  
  1. Damar iskeleti çıkarıldı.  
  2. Beyaz piksel sayısı ile toplam damar uzunluğu bulundu.  
  3. Konturların çevresi hesaplandı.  
  4. Çevre uzunluğu / damar uzunluğu oranı ile kıvrıklık derecesi elde edildi.  
- **Kenar Yoğunluğu:** DR'li gözlerde eksüda, mikroanevrizma ve kanama gibi yüksek kontrastlı yapılar fazla kenar bilgisi ürettiğinden kenar yoğunluğu artar.  
- **Yoğunluk Standart Sapması:** DR’li gözlerde heterojenlik arttığı için standart sapma yükselir.

> Her görüntü için 4 farklı segmentasyondan toplam 8 öznitelik çıkarılmış ve **`train_features.csv`** ile **`test_features.csv`** dosyalarına kaydedilmiştir.

---

## 📊 Makine Öğrenmesi ve Normalizasyon
- Özellikler **StandardScaler** ile normalize edilmiştir (ortalama=0, std=1).  
- Sınıflandırma modelleri:
  - **Random Forest**
  - **Support Vector Machines (SVM)**
  - **Logistic Regression**
  - **K-Nearest Neighbors (KNN)**

### Değerlendirme Metrikleri
- Accuracy (Doğruluk)
- Precision
- F1-Score
- Confusion Matrix

---

## 🏆 Model Sonuçları
- **Random Forest**: Birden fazla karar ağacının birleşiminden oluşan ve aşırı öğrenmeye karşı dayanıklı ensemble algoritması.  
  - Hiperparametre optimizasyonu **GridSearchCV** ile 5 katlı çapraz doğrulama (5-fold CV) kullanılarak yapılmıştır.  
  - En iyi parametreler seçildikten sonra test veri setinde en yüksek başarı elde edilmiştir.  

- **SVM**: Verileri en iyi ayıran karar sınırlarını bulur; GridSearchCV ile optimize edilmiştir.

- **Logistic Regression**: Doğrusal ayrılabilen durumlar için etkili, yorumlanabilir bir yöntem.

- **KNN**: Bir örneği, en yakın k komşusunun sınıfına göre belirler; farklı parametrelerle esnek çalışır.

**En iyi sonucu veren model: Random Forest**

---

## 📷 Görseller
> Buradaki görselleri siz ekleyeceksiniz. Önerilen yer tutucular:

- Segmentasyon örnekleri (optik disk, damar, mikroanevrizma, eksüda)  
  `<img width="385" height="280" alt="image" src="https://github.com/user-attachments/assets/c5cc029a-3fa2-4bd3-a91d-f69a391985bb" />
  
`  
  `<img width="423" height="281" alt="image" src="https://github.com/user-attachments/assets/e6c131b3-a1e1-4c8c-84a2-39aa3e729e70" />
  
`  
  `<img width="386" height="256" alt="image" src="https://github.com/user-attachments/assets/370c90b2-820b-45dc-a22c-97f6aa4ca8be" />
  
`  
  `<img width="436" height="255" alt="image" src="https://github.com/user-attachments/assets/3bebff95-63aa-4b3c-bbba-4a10c3557f31" />
  
`

- CSV önizlemeleri
  Tablo1: Train_features.csv tablosu:
  
  `<img width="1015" height="342" alt="image" src="https://github.com/user-attachments/assets/aebdfc6c-98a9-4f61-b12d-e91743524600" />
  
`  
  Tablo2: Test_features.csv tablosu

  `<img width="1038" height="379" alt="image" src="https://github.com/user-attachments/assets/a821f659-6a22-4dc8-9f52-33ae41e8ca63" />
  
`


- Sonuç grafikleri ve Confusion Matrix
   Tablo 3: Random forest sonuç tablosu
  `<img width="586" height="239" alt="image" src="https://github.com/user-attachments/assets/6665629b-9512-4daa-b24a-beed5979fac4" />
  
`  
'<img width="725" height="604" alt="image" src="https://github.com/user-attachments/assets/01a88449-b0ce-4c98-82a0-54d7aebf4583" />

'

Bu görselleri `images/` klasörüne yerleştirdikten sonra yukarıdaki markdown bağlantılarını güncellemeniz yeterlidir.

---

## 📂 Veri Seti
- **IDRiD (Indian Diabetic Retinopathy Image Dataset)**
  - Eğitim: 100 DR + 100 Normal
  - Test: 50 DR + 50 Normal
- Her görüntü 512×512 RGB formatında.

---

## 🔧 Kurulum
```bash
git clone <repo-url>
cd <repo-folder>

# Sanal ortam (opsiyonel)
python -m venv venv
source venv/bin/activate  # Windows için: venv\Scripts\activate

# Bağımlılıklar
pip install -r requirements.txt
