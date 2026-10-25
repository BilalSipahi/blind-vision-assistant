package com.example.blindvision

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min

class YoloDetector(
    context: Context,
    modelPath: String = "yolov8n_float16.tflite",
    labelPath: String = "labels.txt"
) {
    private var interpreter: Interpreter
    private var labels: List<String>
    private val inputSize = 640
    private val numElements = 8400
    private val numChannels = 84
    private val confThreshold = 0.45f
    private val iouThreshold = 0.50f

    init {
        val options = Interpreter.Options()
        val compatList = CompatibilityList()
        if (compatList.isDelegateSupportedOnThisDevice) {
            options.addDelegate(GpuDelegate(compatList.bestOptionsForThisDevice))
        } else {
            options.setNumThreads(4)
        }
        val modelBuffer = FileUtil.loadMappedFile(context, modelPath)
        interpreter = Interpreter(modelBuffer, options)
        labels = FileUtil.loadLabels(context, labelPath)
    }

    fun detect(bitmap: Bitmap): List<DetectionResult> {
        val resized = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, false)
        val inputBuffer = bitmapToByteBuffer(resized)

        // YOLOv8 TFLite cikti boyutu: [1, 84, 8400]
        val output = Array(1) { Array(numChannels) { FloatArray(numElements) } }
        interpreter.run(inputBuffer, output)

        val candidates = mutableListOf<BoxCandidate>()
        for (i in 0 until numElements) {
            var maxConf = -1f
            var maxClassIdx = -1
            for (c in 4 until numChannels) {
                val score = output[0][c][i]
                if (score > maxConf) {
                    maxConf = score
                    maxClassIdx = c - 4
                }
            }

            if (maxConf >= confThreshold && maxClassIdx in labels.indices) {
                val cx = output[0][0][i]
                val cy = output[0][1][i]
                val w = output[0][2][i]
                val h = output[0][3][i]

                val normCx = if (cx > 1f) cx / inputSize else cx
                val normCy = if (cy > 1f) cy / inputSize else cy
                val normW = if (w > 1f) w / inputSize else w
                val normH = if (h > 1f) h / inputSize else h

                candidates.add(
                    BoxCandidate(
                        label = labels[maxClassIdx],
                        conf = maxConf,
                        cx = normCx,
                        cy = normCy,
                        w = normW,
                        h = normH
                    )
                )
            }
        }

        return applyNMS(candidates).map {
            DetectionResult(
                label = it.label,
                confidence = it.conf,
                normalizedCenterX = it.cx,
                normalizedCenterY = it.cy,
                boxAreaRatio = it.w * it.h
            )
        }
    }

    private fun bitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3)
        buffer.order(ByteOrder.nativeOrder())
        val intValues = IntArray(inputSize * inputSize)
        bitmap.getPixels(intValues, 0, inputSize, 0, 0, inputSize, inputSize)

        for (pixel in intValues) {
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f
            buffer.putFloat(r)
            buffer.putFloat(g)
            buffer.putFloat(b)
        }
        buffer.rewind()
        return buffer
    }

    private fun applyNMS(boxes: List<BoxCandidate>): List<BoxCandidate> {
        val sorted = boxes.sortedByDescending { it.conf }.toMutableList()
        val selected = mutableListOf<BoxCandidate>()

        while (sorted.isNotEmpty()) {
            val current = sorted.removeAt(0)
            selected.add(current)
            val iterator = sorted.iterator()
            while (iterator.hasNext()) {
                val next = iterator.next()
                if (calculateIoU(current, next) > iouThreshold) {
                    iterator.remove()
                }
            }
        }
        return selected
    }

    private fun calculateIoU(a: BoxCandidate, b: BoxCandidate): Float {
        val x1 = max(a.cx - a.w / 2f, b.cx - b.w / 2f)
        val y1 = max(a.cy - a.h / 2f, b.cy - b.h / 2f)
        val x2 = min(a.cx + a.w / 2f, b.cx + b.w / 2f)
        val y2 = min(a.cy + a.h / 2f, b.cy + b.h / 2f)

        val interArea = max(0f, x2 - x1) * max(0f, y2 - y1)
        val unionArea = (a.w * a.h) + (b.w * b.h) - interArea
        return if (unionArea <= 0f) 0f else interArea / unionArea
    }

    fun close() {
        interpreter.close()
    }

    private data class BoxCandidate(
        val label: String,
        val conf: Float,
        val cx: Float,
        val cy: Float,
        val w: Float,
        val h: Float
    )
}
