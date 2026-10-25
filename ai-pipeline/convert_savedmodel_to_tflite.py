import os
import tensorflow as tf

saved_model_dir = "yolov8n_saved_model"
print(f"[1/2] {saved_model_dir} SavedModel TFLite formatina (Float16) donusturuluyor...")

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

assets_dir = os.path.join("..", "android-app", "app", "src", "main", "assets")
os.makedirs(assets_dir, exist_ok=True)

target_path = os.path.join(assets_dir, "yolov8n_float16.tflite")
with open(target_path, "wb") as f:
    f.write(tflite_model)

size_mb = os.path.getsize(target_path) / (1024 * 1024)
print(f"[2/2] Model basariyla kaydedildi: {target_path} ({size_mb:.2f} MB)")
