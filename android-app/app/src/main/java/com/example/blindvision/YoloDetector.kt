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
    private var numElements = 8400
    private var numChannels = 84
    private var isTransposed = false
    private var isNCHW = false
    private val confThreshold = 0.20f // Tespit hassasiyeti (Kucuk objeler / su sisesi icin 0.20)
    private val iouThreshold = 0.45f

    private val inputBuffer: ByteBuffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3).apply {
        order(ByteOrder.nativeOrder())
    }
    private val intValues = IntArray(inputSize * inputSize)

    private var outputNotTransposed: Array<Array<FloatArray>>? = null
    private var outputTransposed: Array<Array<FloatArray>>? = null

    init {
        val options = Interpreter.Options()
        try {
            val compatList = CompatibilityList()
            if (compatList.isDelegateSupportedOnThisDevice) {
                options.addDelegate(GpuDelegate(compatList.bestOptionsForThisDevice))
            } else {
                options.setNumThreads(4)
            }
        } catch (_: Exception) {
            options.setNumThreads(4)
        }

        val modelBuffer = FileUtil.loadMappedFile(context, modelPath)
        interpreter = Interpreter(modelBuffer, options)
        labels = FileUtil.loadLabels(context, labelPath)

        val inputShape = interpreter.getInputTensor(0).shape()
        // [1, 3, 640, 640] -> NCHW format
        isNCHW = (inputShape.size == 4 && inputShape[1] == 3)

        val outputShape = interpreter.getOutputTensor(0).shape()
        if (outputShape.size == 3) {
            if (outputShape[1] == 84 || outputShape[1] < outputShape[2]) {
                numChannels = outputShape[1]
                numElements = outputShape[2]
                isTransposed = false
            } else {
                numElements = outputShape[1]
                numChannels = outputShape[2]
                isTransposed = true
            }
        }

        if (!isTransposed) {
            outputNotTransposed = Array(1) { Array(numChannels) { FloatArray(numElements) } }
        } else {
            outputTransposed = Array(1) { Array(numElements) { FloatArray(numChannels) } }
        }
    }

    fun detect(bitmap: Bitmap): List<DetectionResult> {
        val resized = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, false)
        val inBuf = bitmapToByteBuffer(resized)

        val candidates = mutableListOf<BoxCandidate>()

        if (!isTransposed) {
            val output = outputNotTransposed!!
            interpreter.run(inBuf, output)

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
        } else {
            val output = outputTransposed!!
            interpreter.run(inBuf, output)

            for (i in 0 until numElements) {
                var maxConf = -1f
                var maxClassIdx = -1
                for (c in 4 until numChannels) {
                    val score = output[0][i][c]
                    if (score > maxConf) {
                        maxConf = score
                        maxClassIdx = c - 4
                    }
                }

                if (maxConf >= confThreshold && maxClassIdx in labels.indices) {
                    val cx = output[0][i][0]
                    val cy = output[0][i][1]
                    val w = output[0][i][2]
                    val h = output[0][i][3]

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
        inputBuffer.rewind()
        bitmap.getPixels(intValues, 0, inputSize, 0, 0, inputSize, inputSize)

        val totalPixels = inputSize * inputSize

        if (isNCHW) {
            // NCHW format [1, 3, 640, 640]: Red plane, Green plane, Blue plane
            for (i in 0 until totalPixels) {
                val pixel = intValues[i]
                inputBuffer.putFloat(((pixel shr 16) and 0xFF) / 255.0f)
            }
            for (i in 0 until totalPixels) {
                val pixel = intValues[i]
                inputBuffer.putFloat(((pixel shr 8) and 0xFF) / 255.0f)
            }
            for (i in 0 until totalPixels) {
                val pixel = intValues[i]
                inputBuffer.putFloat((pixel and 0xFF) / 255.0f)
            }
        } else {
            // NHWC format [1, 640, 640, 3]: Interleaved R, G, B
            for (i in 0 until totalPixels) {
                val pixel = intValues[i]
                inputBuffer.putFloat(((pixel shr 16) and 0xFF) / 255.0f)
                inputBuffer.putFloat(((pixel shr 8) and 0xFF) / 255.0f)
                inputBuffer.putFloat((pixel and 0xFF) / 255.0f)
            }
        }

        inputBuffer.rewind()
        return inputBuffer
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
        try {
            interpreter.close()
        } catch (_: Exception) {}
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
