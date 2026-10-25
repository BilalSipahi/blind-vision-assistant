package com.example.blindvision

import android.os.SystemClock
import java.util.Locale

data class DetectionResult(
    val label: String,
    val confidence: Float,
    val normalizedCenterX: Float,
    val normalizedCenterY: Float,
    val boxAreaRatio: Float
)

class DecisionEngine {
    private val lastSpokenTimestamps = HashMap<String, Long>()
    private val cooldownMs = 3500L

    private val labelTranslations = mapOf(
        "person" to "insan", "insan" to "insan",
        "car" to "araba", "araba" to "araba",
        "motorcycle" to "motosiklet", "motosiklet" to "motosiklet",
        "bus" to "otobüs", "otobus" to "otobüs",
        "truck" to "kamyon", "kamyon" to "kamyon",
        "bicycle" to "bisiklet", "bisiklet" to "bisiklet",
        "traffic light" to "trafik ışığı", "trafik isigi" to "trafik ışığı",
        "stop sign" to "dur tabelası", "dur tabelasi" to "dur tabelası",
        "bench" to "bank", "bank" to "bank",
        "chair" to "sandalye", "sandalye" to "sandalye",
        "couch" to "koltuk", "koltuk" to "koltuk",
        "dog" to "köpek", "kopek" to "köpek",
        "cat" to "kedi", "kedi" to "kedi",
        "bottle" to "şişe", "sise" to "şişe",
        "cup" to "bardak", "bardak" to "bardak",
        "dining table" to "masa", "yemek masasi" to "masa",
        "cell phone" to "telefon", "cep telefonu" to "telefon",
        "laptop" to "bilgisayar", "dizustu bilgisayar" to "bilgisayar",
        "backpack" to "sırt çantası", "sirt cantasi" to "sırt çantası",
        "suitcase" to "bavul", "bavul" to "bavul",
        "umbrella" to "şemsiye", "semsiye" to "şemsiye",
        "potted plant" to "saksı bitkisi", "saksi bitkisi" to "saksı bitkisi",
        "book" to "kitap", "kitap" to "kitap",
        "tv" to "televizyon", "televizyon" to "televizyon",
        "keyboard" to "klavye", "klavye" to "klavye",
        "mouse" to "fare", "fare" to "fare"
    )

    fun getDisplayLabel(rawLabel: String): String {
        val key = rawLabel.lowercase()
        return labelTranslations[key] ?: rawLabel
    }

    fun calculateClockDirection(normX: Float): String {
        return when {
            normX < 0.35f -> "Saat 10 yönünde"
            normX > 0.65f -> "Saat 2 yönünde"
            else -> "Önünde"
        }
    }

    fun evaluateDetections(detections: List<DetectionResult>, tofDistanceCm: Int): String? {
        if (detections.isEmpty()) return null

        val relevant = detections.maxByOrNull { it.confidence * it.boxAreaRatio } ?: return null

        val direction = calculateClockDirection(relevant.normalizedCenterX)

        // Merkezdeki cisim icin ToF sensorunu kullan, kenardaki cisim icin bounding-box tahmini yap
        val estimatedCm = if (relevant.normalizedCenterX in 0.35f..0.65f && tofDistanceCm in 10..350) {
            tofDistanceCm
        } else {
            estimateDistanceFromBox(relevant.boxAreaRatio)
        }

        if (estimatedCm > 300 || estimatedCm <= 0) return null

        val displayLabel = getDisplayLabel(relevant.label)

        val now = SystemClock.elapsedRealtime()
        val key = "${displayLabel}_$direction"
        val lastTime = lastSpokenTimestamps[key] ?: 0L

        if (now - lastTime < cooldownMs) return null

        lastSpokenTimestamps[key] = now
        val meters = estimatedCm / 100.0f
        return String.format(Locale.getDefault(), "%s, %.1f metrede %s", direction, meters, displayLabel)
    }

    private fun estimateDistanceFromBox(areaRatio: Float): Int {
        return when {
            areaRatio > 0.35f -> 50
            areaRatio > 0.15f -> 80
            areaRatio > 0.05f -> 120
            areaRatio > 0.015f -> 180
            else -> 250
        }
    }
}
