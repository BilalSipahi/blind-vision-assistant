package com.example.blindvision

import android.os.SystemClock

data class DetectionResult(
    val label: String,
    val confidence: Float,
    val normalizedCenterX: Float,
    val normalizedCenterY: Float,
    val boxAreaRatio: Float
)

class DecisionEngine {
    private val lastSpokenTimestamps = HashMap<String, Long>()
    private val cooldownMs = 4000L

    private val priorityLabels = setOf(
        "insan", "araba", "motosiklet", "otobus", "kamyon", "bisiklet",
        "trafik isigi", "dur tabelasi", "bank", "sandalye", "koltuk", "kopek"
    )

    fun calculateClockDirection(normX: Float): String {
        return when {
            normX < 0.35f -> "Saat 10 yönünde"
            normX > 0.65f -> "Saat 2 yönünde"
            else -> "Önünde"
        }
    }

    fun evaluateDetections(detections: List<DetectionResult>, tofDistanceCm: Int): String? {
        val relevant = detections
            .filter { it.label in priorityLabels }
            .maxByOrNull { it.boxAreaRatio } ?: return null

        val direction = calculateClockDirection(relevant.normalizedCenterX)

        // Merkezdeki cisim icin ToF sensorunu kullan, kenardaki cisim icin bounding-box tahmini yap
        val estimatedCm = if (relevant.normalizedCenterX in 0.35f..0.65f && tofDistanceCm in 10..350) {
            tofDistanceCm
        } else {
            estimateDistanceFromBox(relevant.boxAreaRatio)
        }

        if (estimatedCm > 220 || estimatedCm <= 0) return null

        val now = SystemClock.elapsedRealtime()
        val key = "${relevant.label}_$direction"
        val lastTime = lastSpokenTimestamps[key] ?: 0L

        if (now - lastTime < cooldownMs) return null

        lastSpokenTimestamps[key] = now
        val meters = estimatedCm / 100.0f
        return String.format("%s, %.1f metrede %s", direction, meters, relevant.label)
    }

    private fun estimateDistanceFromBox(areaRatio: Float): Int {
        return when {
            areaRatio > 0.40f -> 80
            areaRatio > 0.20f -> 140
            areaRatio > 0.08f -> 200
            else -> 300
        }
    }
}
