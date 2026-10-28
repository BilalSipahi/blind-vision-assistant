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
    private val lastSpokenMap = HashMap<String, Long>()
    private val cooldownMs = 3500L
    private var lastSpokenAnnouncement: String? = null

    // Anlamlı ve geçerli nesne isimleri sözlüğü (Bilinmeyen veya anlamsız kelimeler kesinlikle seslendirilmez)
    private val validMeaningfulObjects = mapOf(
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
        "couch" to "koltuk", "koltuk" to "koltuk", "sofa" to "koltuk",
        "dog" to "köpek", "kopek" to "köpek",
        "cat" to "kedi", "kedi" to "kedi",
        "bottle" to "şişe", "sise" to "şişe",
        "cup" to "bardak", "bardak" to "bardak",
        "dining table" to "masa", "yemek masasi" to "masa", "masa" to "masa",
        "cell phone" to "telefon", "cep telefonu" to "telefon", "telefon" to "telefon",
        "laptop" to "bilgisayar", "dizustu bilgisayar" to "bilgisayar", "bilgisayar" to "bilgisayar",
        "backpack" to "sırt çantası", "sirt cantasi" to "sırt çantası", "çanta" to "çanta",
        "suitcase" to "bavul", "bavul" to "bavul",
        "umbrella" to "şemsiye", "semsiye" to "şemsiye",
        "potted plant" to "saksı bitkisi", "saksi bitkisi" to "saksı bitkisi",
        "book" to "kitap", "kitap" to "kitap",
        "tv" to "televizyon", "televizyon" to "televizyon",
        "keyboard" to "klavye", "klavye" to "klavye",
        "mouse" to "fare", "fare" to "fare",
        "door" to "kapı", "kapi" to "kapı",
        "clock" to "saat", "saat" to "saat",
        "vase" to "vazo", "vazo" to "vazo",
        "scissors" to "makas", "makas" to "makas",
        "teddy bear" to "oyuncak ayı", "oyuncak ayi" to "oyuncak ayı"
    )

    fun getDisplayLabel(rawLabel: String): String? {
        val key = rawLabel.lowercase().trim()
        return validMeaningfulObjects[key]
    }

    fun calculateDirection(normX: Float): String {
        return when {
            normX < 0.33f -> "Solunda"
            normX > 0.66f -> "Sağında"
            else -> "Önünde"
        }
    }

    /**
     * Kadrajdaki tespitleri değerlendirir:
     * 1. Şartlama: Yalnızca anlamlı nesne sözlüğünde olan nesneler kabul edilir.
     * 2. En Fazla Yer Kaplayan Cisim: Kamerada bounding box alanı (boxAreaRatio) EN BÜYÜK olan tek nesne seçilir.
     * 3. Cooldown: Aynı nesne 3.5 sn geçmeden tekrar anons edilmez.
     */
    fun evaluateDetections(detections: List<DetectionResult>, tofDistanceCm: Int): String? {
        if (detections.isEmpty()) return null

        // 1. Şartlama: Sadece anlamlı bir Türkçe nesne adına karşılık gelen nesneleri al
        val validDetections = detections.mapNotNull { detection ->
            val turkishLabel = getDisplayLabel(detection.label)
            if (turkishLabel != null) {
                Pair(detection, turkishLabel)
            } else {
                null
            }
        }

        // Anlamlı bir kelime/nesne yoksa KESİNLİKLE SESLENDİRME YAPMA
        if (validDetections.isEmpty()) return null

        // 2. Kamerada EN FAZLA YER KAPLAYAN (boxAreaRatio en büyük olan) cismi seç
        val (largestObject, displayLabel) = validDetections.maxByOrNull { it.first.boxAreaRatio } ?: return null

        val now = SystemClock.elapsedRealtime()

        // 3. Cooldown Kontrolü (3.5 saniye)
        val lastSpokenTime = lastSpokenMap[displayLabel] ?: 0L
        if (now - lastSpokenTime < cooldownMs) {
            return null
        }

        val direction = calculateDirection(largestObject.normalizedCenterX)

        val estimatedCm = if (largestObject.normalizedCenterX in 0.33f..0.66f && tofDistanceCm in 10..350) {
            tofDistanceCm
        } else {
            estimateDistanceFromBox(largestObject.boxAreaRatio)
        }

        if (estimatedCm > 300 || estimatedCm <= 0) return null

        val meters = estimatedCm / 100.0f
        val announcement = String.format(Locale.getDefault(), "%s, %.1f metrede %s", direction, meters, displayLabel)

        // Gereksiz tekrarları önleme
        if (announcement == lastSpokenAnnouncement && (now - lastSpokenTime < cooldownMs * 2)) {
            return null
        }

        lastSpokenMap[displayLabel] = now
        lastSpokenAnnouncement = announcement

        return announcement
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
