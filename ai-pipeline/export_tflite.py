import os
import shutil
from ultralytics import YOLO

COCO_LABELS_TR = [
    "insan", "bisiklet", "araba", "motosiklet", "ucak", "otobus", "tren", "kamyon", "tekne",
    "trafik isigi", "yangin muslugu", "dur tabelasi", "parkmetre", "bank", "kus", "kedi",
    "kopek", "at", "koyun", "inek", "fil", "ayi", "zebra", "zurafa", "sirt cantasi",
    "semsiye", "el cantasi", "kravat", "bavul", "frizbi", "kayak", "snowboard", "spor topu",
    "ucurtma", "beyzbol sopasi", "beyzbol eldiveni", "kaykay", "sorf tahtasi", "tenis raketi",
    "sise", "sarap kadehi", "bardak", "catal", "bicak", "kasik", "kase", "muz", "elma",
    "sandvic", "portakal", "brokoli", "havuc", "sosisli", "pizza", "corek", "pasta",
    "sandalye", "koltuk", "saksi bitkisi", "yatak", "yemek masasi", "tuvalet", "televizyon",
    "dizustu bilgisayar", "fare", "uzaktan kumanda", "klavye", "cep telefonu", "mikrodalga",
    "firin", "ekmek kizartma", "lavabo", "buzdolabi", "kitap", "saat", "vazo", "makas",
    "oyuncak ayi", "sac kurutma", "dis fircasi"
]

def main():
    print("[1/3] YOLOv8n modeli indiriliyor...")
    model = YOLO("yolov8n.pt")

    print("[2/3] Model FP16 TFLite formatina donusturuluyor...")
    export_path = model.export(format="tflite", half=True, imgsz=640)

    assets_dir = os.path.join("..", "android-app", "app", "src", "main", "assets")
    os.makedirs(assets_dir, exist_ok=True)

    target_tflite = os.path.join(assets_dir, "yolov8n_float16.tflite")
    if os.path.exists(str(export_path)):
        shutil.copy(str(export_path), target_tflite)
        print(f"Model kopyalandi: {target_tflite}")

    print("[3/3] Turkce etiket dosyasi (labels.txt) olusturuluyor...")
    labels_path = os.path.join(assets_dir, "labels.txt")
    with open(labels_path, "w", encoding="utf-8") as f:
        for label in COCO_LABELS_TR:
            f.write(f"{label}\n")
    print(f"Etiketler kaydedildi: {labels_path}")

if __name__ == "__main__":
    main()
