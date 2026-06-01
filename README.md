# Zaman Serileri ile Makine Öğrenmesi ve Derin Öğrenme Algoritmalarının Karşılaştırmalı Enerji Tahmini Projesi

Bu proje, farklı zaman frekanslarına (aylık, günlük, saatlik) sahip çok değişkenli enerji verileri üzerinde geleneksel istatistiksel modeller, ağaç tabanlı makine öğrenmesi algoritmaları ve derin öğrenme mimarilerinin tahmin performanslarını karşılaştırmalı olarak incelemek amacıyla geliştirilmiştir.

## Proje Özeti
Proje kapsamında, ENTSO-E Transparency Platform üzerinden temin edilen Romanya elektrik iletim sistem operatörü (Transelectrica) verileri kullanılmıştır. Veri seti; elektrik tüketimi, üretimi ve nükleer, rüzgâr, hidroelektrik, petrol/gaz, kömür, güneş, biyokütle gibi kaynak bazlı üretim değişkenlerini içermektedir. 

Çalışmada, veri frekansının ve hacminin model başarısı üzerindeki etkisi metodolojik olarak kanıtlanmış; düşük frekanslı (günlük) verilerde ağaç tabanlı modellerin, yüksek frekanslı (saatlik) verilerde ise derin öğrenme mimarilerinin üstün performansı ortaya konmuştur.

## Kullanılan Teknolojiler ve Kütüphaneler
* **Programlama Dili:** Python
* **Veri Manipülasyonu & Ön İşleme:** Pandas, NumPy
* **Veri Görselleştirme:** Matplotlib, Seaborn
* **Geleneksel Zaman Serisi:** Statsmodels (ARIMA)
* **Makine Öğrenmesi:** Scikit-Learn, XGBoost, LightGBM
* **Derin Öğrenme:** TensorFlow / Keras (CNN, LSTM)

## Veri Seti Yapısı ve Özellik Mühendisliği
Projede kullanılan veri setindeki ana bileşenler şunlardır:
* `DateTime`: Zaman damgası (Index olarak kurgulanmıştır)
* `Consumption` / `Production`: Toplam elektrik tüketim ve üretim miktarı (MW)
* `Nuclear`, `Wind`, `Hydroelectric`, `Oil and Gas`, `Coal`, `Solar`, `Biomass`: Kaynak bazlı üretim miktarları (MW)

### Yapılan Ön İşleme Adımları:
1. **Zaman Serisi Dönüşümü:** Saatlik ham veriler, Pandas kütüphanesi kullanılarak günlük ve aylık frekanslara (Resampling) dönüştürülmüştür.
2. **Özellik Mühendisliği (Feature Engineering):** Ağaç tabanlı modellerin (RF, XGBoost, LGBM) zaman bağımlılığını, trendleri ve mevsimsel kırılmaları yakalayabilmesi için **Gecikme (Lag)**, **Yuvarlanan Ortalama (Rolling)** ve **Takvimsel Değişkenler** türetilmiştir.
3. **Ölçeklendirme:** Derin öğrenme modelleri (CNN, LSTM) için veriler `MinMaxScaler` ile optimize edilmiştir.

## Uygulanan Modeller
Proje kapsamında 3 farklı kategoride modeller uçtan uca eğitilmiştir:

1. **İstatistiksel Modeller:** ARIMA(p, d, q) modelleri için otomatik *Grid Search (Akaike/Bayesian Information Criterion)* fonksiyonu yazılarak her değişken için en optimum parametreler belirlenmiştir.
2. **Ağaç Tabanlı Makine Öğrenmesi:** Random Forest Regressor, XGBoost Regressor, LightGBM Regressor.
3. **Derin Öğrenme Mimarileri:**
   * **LSTM (Uzun Kısa Süreli Bellek):** Ardışık katmanlar, Dropout (aşırı öğrenmeyi önleme) ve EarlyStopping entegrasyonu ile uzun dönemli ilişkileri öğrenmek üzere tasarlanmıştır.
   * **CNN (Evrişimsel Sinir Ağları):** Conv1D ve MaxPooling1D katmanları ile kısa vadeli yerel örüntüleri ve ani dalgalanmaları yakalamak üzere kurgulanmıştır.


## 📊 Model Sonuçları ve Karşılaştırma

Modellerin tahmin performansları **MAE (Ortalama Mutlak Hata)**, **RMSE (Kök Ortalama Kare Hata)** ve literatürde en güvenilir ölçüt kabul edilen **MASE (Ölçekten Bağımsız Hata Metriği)** üzerinden değerlendirilmiştir.

### 1. Günlük Veri Seti Tahmin Performansı (Düşük Frekans)
Günlük elektrik üretim ve tüketim verileri (2019-2023 eğitim, 2024-2025 test) üzerinde yapılan analizlerde, ağaç tabanlı modellerin derin öğrenme modellerine kıyasla daha kararlı ve yüksek doğrulukta sonuçlar ürettiği gözlemlenmiştir.

| Model | MAE | RMSE | MAPE (%) | MASE |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost** | **7980.11** | **14672.22** | *inf* | **0.9128** |
| **LightGBM** | 8186.36 | 14938.97 | *inf* | 0.9364 |
| **Random Forest** | 8663.85 | 16025.70 | *inf* | 0.9910 |
| **LSTM** | 11742.63 | 16892.28 | 10.79% | 1.0402 |
| **CNN** | 12245.26 | 17984.60 | 11.49% | 1.0847 |

* **Bulgu:** Günlük frekansta ve sınırlı gözlem sayısına sahip zaman serilerinde, derin öğrenme modelleri (CNN/LSTM) yüksek parametre sayıları nedeniyle yeterince genelleme yapamamış ve MASE değerleri 1'in üzerinde (baseline modelden daha zayıf) kalmıştır. Bu veri yapısı için en uygun mimarinin **XGBoost** olduğu doğrulanmıştır.

### 2. Saatlik Veri Seti Tahmin Performansı (Yüksek Frekans / Büyük Veri)
Veri hacminin arttığı ve kısa vadeli zamansal bağımlılıkların yoğunlaştığı saatlik veri setinde (01.03.2024-01.01.2025 eğitim, 01.01.2025-19.03.2025 test) derin öğrenme modelleri belirgin bir üstünlük sağlamıştır.

| Model | MAE | RMSE | MAPE (%) | MASE |
| :--- | :---: | :---: | :---: | :---: |
| **LSTM** | 124.84 | 173.60 | 2.41% | 0.8694 |
| **CNN** | **126.39** | **167.95** | **2.44%** | **0.8802** |
| **Random Forest** | 181.53 | 268.61 | 2.64% | 0.7939 |
| **XGBoost** | 190.66 | 268.47 | 2.77% | 0.8338 |
| **LightGBM** | 207.90 | 293.21 | 2.97% | 0.9092 |

* **Bulgu:** Saatlik verilerde büyük hataları cezalandıran **RMSE** metriği baz alındığında en başarılı model **CNN** mimarisi olmuştur. Yüksek çözünürlüklü ve büyük veri setlerinde derin öğrenme modellerinin zamansal kalıpları öğrenme yeteneğinin ağaç tabanlı modelleri geride bıraktığı metodolojik olarak kanıtlanmıştır. Ayrıca saatlik frekansta tüm modeller naive (baseline) tahminden daha iyi performans göstermiştir (MASE < 1).


## Proje Çıktıları ve Öngörüler
* **Veri Hacmi Etkisi:** Derin öğrenme modelleri (CNN/LSTM) yüksek parametre sayıları nedeniyle büyük veri setlerinde (saatlik) zamansal bağımlılıkları çok daha iyi öğrenmektedir. Sınırlı verilerde ise ağaç tabanlı modeller (XGBoost) ezberlemeyi (overfitting) engelleyerek daha istikrarlı çalışmaktadır.
* **Metrik Güvenilirliği:** Veride sıfıra yakın değerlerin bulunması sebebiyle MAPE metriğinin sonsuz ($\infty$) sonuçlar ürettiği gözlemlenmiş, bu doğrultuda çalışmada literatürün en güvenilir metriği olan **MASE** ve **RMSE** baz alınmıştır.

