package com.example.blindvision

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint
import android.os.SystemClock
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageProxy
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.Text
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import kotlin.math.abs
import kotlin.math.hypot

class TextReader(private val onTextRecognized: (String) -> Unit) {

    private val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
    private var lastSpokenText: String? = null
    private var lastSpokenTime: Long = 0L
    private val cooldownMs = 3500L

    @ExperimentalGetImage
    fun processImage(imageProxy: ImageProxy) {
        val mediaImage = imageProxy.image
        if (mediaImage == null) {
            imageProxy.close()
            return
        }

        val rawBitmap = try { imageProxy.toBitmap() } catch (_: Exception) { null }
        val rotationDegrees = imageProxy.imageInfo.rotationDegrees

        // 1. Standart İşleme
        val image = InputImage.fromMediaImage(mediaImage, rotationDegrees)

        recognizer.process(image)
            .addOnSuccessListener { visionText ->
                // Şişe/Ambalaj üzerindeki şeffaf/stilize fontlar için eğer metin boş çıktıysa Kontrast Artırılmış İşleme yap
                if (visionText.text.isBlank() && rawBitmap != null) {
                    val enhancedBitmap = enhanceBitmapContrast(rawBitmap)
                    val enhancedImage = InputImage.fromBitmap(enhancedBitmap, rotationDegrees)
                    recognizer.process(enhancedImage).addOnSuccessListener { enhancedVisionText ->
                        evaluateAndAnnounceText(enhancedVisionText, rawBitmap)
                    }
                } else {
                    evaluateAndAnnounceText(visionText, rawBitmap)
                }
            }
            .addOnCompleteListener {
                imageProxy.close()
            }
    }

    private fun evaluateAndAnnounceText(visionText: Text, bitmap: Bitmap?) {
        if (visionText.text.isBlank()) return

        val fingertip = if (bitmap != null) detectFingertip(bitmap) else null

        val selectedText = if (fingertip != null) {
            findTextNearFingertip(visionText, fingertip)
        } else {
            findCentralText(visionText)
        }

        if (!selectedText.isNullOrBlank()) {
            // Tek harfleri, parazitleri ve anlamsız OCR çıktılarını temizle
            val validatedText = cleanAndValidateText(selectedText)
            if (validatedText != null) {
                val now = SystemClock.elapsedRealtime()
                if (validatedText != lastSpokenText || (now - lastSpokenTime > cooldownMs)) {
                    lastSpokenText = validatedText
                    lastSpokenTime = now
                    onTextRecognized(validatedText)
                }
            }
        }
    }

    /**
     * Şişe ve şeffaf ambalaj üzerindeki düşük kontrastlı / stilize fontların okunmasını sağlayan görüntü keskinleştirme
     */
    private fun enhanceBitmapContrast(src: Bitmap): Bitmap {
        val cm = ColorMatrix(floatArrayOf(
            1.5f, 0f, 0f, 0f, -25f,
            0f, 1.5f, 0f, 0f, -25f,
            0f, 0f, 1.5f, 0f, -25f,
            0f, 0f, 0f, 1f, 0f
        ))
        val dest = Bitmap.createBitmap(src.width, src.height, src.config ?: Bitmap.Config.ARGB_8888)
        val canvas = Canvas(dest)
        val paint = Paint().apply {
            colorFilter = ColorMatrixColorFilter(cm)
        }
        canvas.drawBitmap(src, 0f, 0f, paint)
        return dest
    }

    /**
     * Tek harfleri, gürültüleri ve anlamsız OCR çıktılarını temizler.
     * Yalnızca en az 2 karakterli anlamlı kelimeleri kabul eder.
     */
    private fun cleanAndValidateText(text: String): String? {
        val clean = text.replace("[^a-zA-Z0-9çğıöşüÇĞİÖŞÜ\\s-]".toRegex(), " ")
        val words = clean.split("\\s+".toRegex()).filter { word ->
            // Tek harfli gürültüleri ele (a, b, x, 1 vb.)
            if (word.length < 2) return@filter false
            // Sadece rakam ise en az 2 haneli olmalı (örn: "15", "100")
            if (word.all { it.isDigit() }) return@filter word.length >= 2
            true
        }

        val result = words.joinToString(" ").trim()
        return if (result.length >= 2) result else null
    }

    private fun detectFingertip(bitmap: Bitmap): Pair<Int, Int>? {
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        var minPointY = height
        var fingertipX = -1
        var skinCount = 0

        val step = 4
        for (y in (height / 3) until height step step) {
            for (x in 0 until width step step) {
                val pixel = pixels[y * width + x]
                val r = (pixel shr 16) and 0xFF
                val g = (pixel shr 8) and 0xFF
                val b = pixel and 0xFF

                if (r > 95 && g > 40 && b > 20 &&
                    (maxOf(r, g, b) - minOf(r, g, b) > 15) &&
                    abs(r - g) > 15 && r > g && r > b) {

                    skinCount++
                    if (y < minPointY) {
                        minPointY = y
                        fingertipX = x
                    }
                }
            }
        }

        return if (skinCount > 150 && fingertipX != -1) {
            Pair(fingertipX, minPointY)
        } else {
            null
        }
    }

    private fun findTextNearFingertip(visionText: Text, fingertip: Pair<Int, Int>): String? {
        var minDistance = Float.MAX_VALUE
        var closestText: String? = null
        val (fx, fy) = fingertip

        for (block in visionText.textBlocks) {
            for (line in block.lines) {
                val box = line.boundingBox ?: continue
                val dist = hypot((fx - box.centerX()).toDouble(), (fy - box.centerY()).toDouble()).toFloat()
                if (dist < minDistance && dist < 250f) {
                    minDistance = dist
                    closestText = line.text
                }
            }
        }
        return closestText
    }

    private fun findCentralText(visionText: Text): String? {
        var bestLine: String? = null
        var minCenterDist = Float.MAX_VALUE

        for (block in visionText.textBlocks) {
            for (line in block.lines) {
                val box = line.boundingBox ?: continue
                val dist = hypot((box.centerX() - 320).toDouble(), (box.centerY() - 320).toDouble()).toFloat()
                if (dist < minCenterDist) {
                    minCenterDist = dist
                    bestLine = line.text
                }
            }
        }
        return bestLine ?: visionText.textBlocks.firstOrNull()?.text
    }

    fun close() {
        try {
            recognizer.close()
        } catch (_: Exception) {}
    }
}
