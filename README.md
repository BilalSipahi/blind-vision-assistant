# Blind Vision Assistant (Görme Engelli Çevre Analiz Sistemi)

Akıllı telefon işlem gücü, geniş açılı UVC/Dahili kamera, ESP32 + VL53L1X ToF lazer mesafe sensörü ve kemik iletimli kulaklık kullanan çevrimdışı (offline) çevre analiz sistemi.

## Sistem Mimarisi ve Ekip Görev Dağılımı
* **`android-app/` (Geliştirici 1):** Kamera akışı, Bluetooth SPP haberleşmesi, TFLite entegrasyonu ve erişilebilir arayüz.
* **`ai-pipeline/` (Geliştirici 2):** YOLOv8n model eğitimi, FP16 TFLite dönüştürme, `YoloDetector` ve `DecisionEngine` algoritmaları.
* **`firmware-esp32/` (Geliştirici 3):** ESP32 + VL53L1X I2C mesafe okuma, hareketli ortalama filtresi ve Bluetooth SPP veri yayını.

## Donanım Pin Bağlantısı (ESP32 - VL53L1X)
| VL53L1X Pin | ESP32 Pin | Açıklama |
| :--- | :--- | :--- |
| VIN | 3V3 | 3.3V Güç Beslemesi |
| GND | GND | Toprak |
| SDA | GPIO 21 | I2C Veri Hattı |
| SCL | GPIO 22 | I2C Saat Hattı |

## Hızlı Kurulum
1. **Model Üretimi:** `cd ai-pipeline && pip install -r requirements.txt && python export_tflite.py`
2. **Model Aktarımı:** Oluşan `yolov8n_float16.tflite` ve `labels.txt` dosyalarını `android-app/app/src/main/assets/` dizinine kopyalayın.
3. **ESP32 Yükleme:** `firmware-esp32` klasörünü PlatformIO ile açıp ESP32 kartına yükleyin.
4. **Mobil Uygulama:** `android-app` klasörünü Android Studio ile açıp derleyin.
