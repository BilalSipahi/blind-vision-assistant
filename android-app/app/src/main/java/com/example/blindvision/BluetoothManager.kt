package com.example.blindvision

import android.annotation.SuppressLint
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothSocket
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import java.io.BufferedReader
import java.io.InputStreamReader
import java.util.UUID

class BluetoothManager(private val onDistanceReceived: (Int) -> Unit) {
    private val sppUuid: UUID = UUID.fromString("00001101-0000-1000-8000-00805F9B34FB")
    private var socket: BluetoothSocket? = null
    @Volatile private var isRunning = false

    @SuppressLint("MissingPermission")
    suspend fun connectAndListen(deviceName: String = "BlindVision_ESP32") = withContext(Dispatchers.IO) {
        isRunning = true
        val adapter = BluetoothAdapter.getDefaultAdapter() ?: return@withContext

        while (isRunning) {
            try {
                val device = adapter.bondedDevices?.find { it.name == deviceName }
                if (device == null) {
                    delay(3000)
                    continue
                }

                socket = device.createRfcommSocketToServiceRecord(sppUuid)
                socket?.connect()

                val reader = BufferedReader(InputStreamReader(socket?.inputStream))
                while (isRunning) {
                    val line = reader.readLine() ?: break
                    if (line.startsWith("DIST:")) {
                        line.substringAfter("DIST:").trim().toIntOrNull()?.let { dist ->
                            onDistanceReceived(dist)
                        }
                    }
                }
            } catch (_: Exception) {
                try { socket?.close() } catch (_: Exception) {}
                socket = null
                delay(2000) // Baglanti koparsa 2 saniye sonra otomatik tekrar dene
            }
        }
    }

    fun close() {
        isRunning = false
        try { socket?.close() } catch (_: Exception) {}
        socket = null
    }
}
