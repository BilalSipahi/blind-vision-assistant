package com.example.blindvision

import android.content.Context
import android.os.SystemClock
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageProxy
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.Text
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions

class CurrencyRecognizer(
    context: Context,
    private val onCurrencyRecognized: (String) -> Unit
) {
    private var currencyDetector: YoloDetector? = null
    private val ocrRecognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)

    private var lastSpokenCurrency: String? = null
    private var lastSpokenTime: Long = 0L
    private val cooldownMs = 3500L

    private val validDenominations = setOf("200", "100", "50", "20", "10", "5")

    init {
        // İleride veya mevcut assets içinde yolov8n_tl.tflite varsa TFLite dedektörü yükle
        try {
            currencyDetector = YoloDetector(context, "yolov8n_tl.tflite", "labels_tl.txt")
        } catch (_: Exception) {
            currencyDetector = null
        }
    }

    @ExperimentalGetImage
    fun processImage(imageProxy: ImageProxy) {
        val mediaImage = imageProxy.image
        if (mediaImage == null) {
            imageProxy.close()
            return
        }

        // 1. TFLite Özel Para Modeli Varsa Onu Çalıştır
        if (currencyDetector != null) {
            try {
                val bitmap = imageProxy.toBitmap()
                val results = currencyDetector!!.detect(bitmap)
                val best = results.maxByOrNull { it.confidence }
                if (best != null && best.confidence >= 0.40f) {
                    val formatted = formatTlLabel(best.label)
                    announceCurrency(formatted)
                    imageProxy.close()
                    return
                }
            } catch (_: Exception) {}
        }

        // 2. Akıllı Sadece TL Kupür Değeri Süzgeci
        val image = InputImage.fromMediaImage(mediaImage, imageProxy.imageInfo.rotationDegrees)
        ocrRecognizer.process(image)
            .addOnSuccessListener { visionText ->
                val detectedTL = parseTLBanknote(visionText)
                if (detectedTL != null) {
                    announceCurrency(detectedTL)
                }
            }
            .addOnCompleteListener {
                imageProxy.close()
            }
    }

    private fun parseTLBanknote(visionText: Text): String? {
        val fullText = visionText.text.uppercase()

        // Türk Lirası banknot emtiabilgisi doğrulaması (TÜRK, LİRA, MERKEZ, BANKASI, TL)
        val hasTlContext = fullText.contains("TÜRK") || fullText.contains("TURK") ||
                fullText.contains("LİRA") || fullText.contains("LIRA") ||
                fullText.contains("MERKEZ") || fullText.contains("BANKASI") ||
                fullText.contains("TL")

        // ML Kit metin blokları ve elemanları arasında tam eşleşen kupür ara
        for (block in visionText.textBlocks) {
            for (line in block.lines) {
                for (element in line.elements) {
                    val cleanToken = element.text.trim().uppercase().replace("[^0-9]".toRegex(), "")
                    if (cleanToken in validDenominations) {
                        return "$cleanToken Türk Lirası"
                    }
                }
            }
        }

        // Standart regex araması (Banknot bağlamı varsa)
        if (hasTlContext) {
            val regex = Regex("\\b(200|100|50|20|10|5)\\b")
            val match = regex.find(fullText)
            if (match != null) {
                return "${match.value} Türk Lirası"
            }
        }

        return null
    }

    private fun formatTlLabel(rawLabel: String): String {
        val clean = rawLabel.replace("[^0-9]".toRegex(), "")
        return if (clean in validDenominations) {
            "$clean Türk Lirası"
        } else {
            rawLabel
        }
    }

    private fun announceCurrency(currencyText: String) {
        val now = SystemClock.elapsedRealtime()
        if (currencyText != lastSpokenCurrency || (now - lastSpokenTime > cooldownMs)) {
            lastSpokenCurrency = currencyText
            lastSpokenTime = now
            onCurrencyRecognized(currencyText)
        }
    }

    fun close() {
        try {
            currencyDetector?.close()
            ocrRecognizer.close()
        } catch (_: Exception) {}
    }
}
