package com.example.blindvision

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Build
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.widget.TextView
import androidx.annotation.OptIn
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch
import java.util.Locale
import java.util.concurrent.Executors

enum class AppMode(val title: String, val spokenAnnouncement: String) {
    OBJECT_DETECTION("Nesne Tespiti Modu", "Nesne tespiti modu devrede."),
    TEXT_READING("Metin Okuma Modu", "Metin okuma modu devrede."),
    CURRENCY_RECOGNITION("Para Tanıma Modu", "Para tanıma modu devrede.")
}

class MainActivity : AppCompatActivity(), TextToSpeech.OnInitListener {

    private lateinit var previewView: PreviewView
    private lateinit var statusText: TextView
    private lateinit var tts: TextToSpeech
    private lateinit var decisionEngine: DecisionEngine
    private lateinit var bluetoothManager: BluetoothManager
    private lateinit var textReader: TextReader
    private lateinit var currencyRecognizer: CurrencyRecognizer
    private var yoloDetector: YoloDetector? = null

    private var currentMode = AppMode.OBJECT_DETECTION

    @Volatile private var currentDistanceCm: Int = -1
    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private var lastInferenceTime = 0L
    private val inferenceIntervalMs = 100L // Saniyede 10 FPS siniri

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val root = androidx.appcompat.widget.LinearLayoutCompat(this).apply {
            orientation = androidx.appcompat.widget.LinearLayoutCompat.VERTICAL
        }

        previewView = PreviewView(this).apply {
            layoutParams = androidx.appcompat.widget.LinearLayoutCompat.LayoutParams(
                androidx.appcompat.widget.LinearLayoutCompat.LayoutParams.MATCH_PARENT,
                0,
                1f
            )
        }

        statusText = TextView(this).apply {
            text = "Sistem Hazırlanıyor..."
            textSize = 20f
            setPadding(32, 32, 32, 32)
        }

        root.addView(previewView)
        root.addView(statusText)
        setContentView(root)

        // Görme engelliler için EKRANA DOKUNMA İLE MOD DEĞİŞTİRME
        root.setOnClickListener {
            switchMode()
        }

        decisionEngine = DecisionEngine()
        tts = TextToSpeech(this, this)

        textReader = TextReader { recognizedText ->
            runOnUiThread {
                statusText.text = "Okunan Metin: $recognizedText"
                speak(recognizedText)
            }
        }

        currencyRecognizer = CurrencyRecognizer(this) { currencyText ->
            runOnUiThread {
                statusText.text = "Algılanan Para: $currencyText"
                speak(currencyText)
            }
        }

        try {
            yoloDetector = YoloDetector(this)
        } catch (e: Exception) {
            statusText.text = "Model Hatası: ${e.localizedMessage}\nassets klasörüne .tflite ve labels.txt ekleyin."
        }

        bluetoothManager = BluetoothManager { dist ->
            currentDistanceCm = dist
        }

        checkAndRequestPermissions()
    }

    private fun switchMode() {
        currentMode = when (currentMode) {
            AppMode.OBJECT_DETECTION -> AppMode.TEXT_READING
            AppMode.TEXT_READING -> AppMode.CURRENCY_RECOGNITION
            AppMode.CURRENCY_RECOGNITION -> AppMode.OBJECT_DETECTION
        }

        statusText.text = currentMode.title
        speak(currentMode.spokenAnnouncement, force = true)
    }

    private fun checkAndRequestPermissions() {
        val perms = mutableListOf(Manifest.permission.CAMERA)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            perms.add(Manifest.permission.BLUETOOTH_CONNECT)
            perms.add(Manifest.permission.BLUETOOTH_SCAN)
        }
        val missing = perms.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
        if (missing.isEmpty()) {
            startSystem()
        } else {
            statusText.text = "Gerekli izinler (Kamera/Bluetooth) bekleniyor..."
            ActivityCompat.requestPermissions(this, missing.toTypedArray(), 101)
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 101) {
            if (grantResults.isNotEmpty() && grantResults.all { it == PackageManager.PERMISSION_GRANTED }) {
                startSystem()
            } else {
                statusText.text = "Kamera ve Bluetooth izinleri verilmedi!"
            }
        }
    }

    private fun startSystem() {
        startCamera()
        lifecycleScope.launch {
            bluetoothManager.connectAndListen()
        }
    }

    @OptIn(ExperimentalGetImage::class)
    private fun startCamera() {
        statusText.text = "Kamera Başlatılıyor..."
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            try {
                val cameraProvider = cameraProviderFuture.get()
                val preview = Preview.Builder().build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

                statusText.text = "${currentMode.title} Aktif"

                val imageAnalysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()

                imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
                    try {
                        val now = System.currentTimeMillis()
                        if (now - lastInferenceTime >= inferenceIntervalMs) {
                            lastInferenceTime = now

                            when (currentMode) {
                                AppMode.OBJECT_DETECTION -> {
                                    if (yoloDetector != null) {
                                        val rawBitmap = imageProxy.toBitmap()
                                        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
                                        val bitmap = if (rotationDegrees != 0) {
                                            val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
                                            Bitmap.createBitmap(rawBitmap, 0, 0, rawBitmap.width, rawBitmap.height, matrix, true)
                                        } else {
                                            rawBitmap
                                        }

                                        val results = yoloDetector!!.detect(bitmap)
                                        val warning = decisionEngine.evaluateDetections(results, currentDistanceCm)

                                        runOnUiThread {
                                            if (warning != null) {
                                                statusText.text = warning
                                                speak(warning)
                                            } else {
                                                val topDetections = results.mapNotNull {
                                                    val label = decisionEngine.getDisplayLabel(it.label)
                                                    if (label != null) "$label (%${(it.confidence * 100).toInt()})" else null
                                                }.take(3).joinToString(", ")

                                                val distanceStr = if (currentDistanceCm > 0) "${currentDistanceCm} cm" else "Bağlı değil"
                                                if (topDetections.isNotEmpty()) {
                                                    statusText.text = "Tespit: $topDetections | Mesafe: $distanceStr"
                                                } else {
                                                    statusText.text = "Sistem Aktif (Tespit Yok) | Sensör: $distanceStr"
                                                }
                                            }
                                        }
                                    }
                                    imageProxy.close()
                                }
                                AppMode.TEXT_READING -> {
                                    // ML Kit Metin Okuma
                                    textReader.processImage(imageProxy)
                                }
                                AppMode.CURRENCY_RECOGNITION -> {
                                    // TL Para Tanıma Modu
                                    currencyRecognizer.processImage(imageProxy)
                                }
                            }
                        } else {
                            imageProxy.close()
                        }
                    } catch (e: Exception) {
                        runOnUiThread {
                            statusText.text = "Hata: ${e.javaClass.name}: ${e.localizedMessage}"
                        }
                        e.printStackTrace()
                        imageProxy.close()
                    }
                }

                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageAnalysis)
            } catch (e: Exception) {
                statusText.text = "Kamera Başlatma Hatası: ${e.localizedMessage}"
            }
        }, ContextCompat.getMainExecutor(this))
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            tts.setLanguage(Locale("tr", "TR"))
            speak("Görme engelli asistanı aktif. Mod değiştirmek için ekrana dokunun.", force = true)
        }
    }

    private fun speak(text: String, force: Boolean = false) {
        if (::tts.isInitialized) {
            if (force) {
                tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, "FORCE_${System.currentTimeMillis()}")
            } else if (!tts.isSpeaking) {
                tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, "ANNOUNCE_${System.currentTimeMillis()}")
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        yoloDetector?.close()
        textReader.close()
        currencyRecognizer.close()
        bluetoothManager.close()
        if (::tts.isInitialized) {
            tts.stop()
            tts.shutdown()
        }
    }
}
