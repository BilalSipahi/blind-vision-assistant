#include <Arduino.h>
#include <Wire.h>
#include <VL53L1X.h>
#include "BluetoothSerial.h"

#define SDA_PIN 21
#define SCL_PIN 22
#define FILTER_SIZE 3

VL53L1X sensor;
BluetoothSerial SerialBT;

int readings[FILTER_SIZE] = {0};
int readIndex = 0;
unsigned long lastSend = 0;
const unsigned long INTERVAL_MS = 66; // ~15 Hz

int getFilteredDistance(int newSample) {
    readings[readIndex] = newSample;
    readIndex = (readIndex + 1) % FILTER_SIZE;
    long sum = 0;
    for (int i = 0; i < FILTER_SIZE; i++) {
        sum += readings[i];
    }
    return (int)(sum / FILTER_SIZE);
}

void setup() {
    Serial.begin(115200);
    Wire.begin(SDA_PIN, SCL_PIN);
    Wire.setClock(400000);

    SerialBT.begin("BlindVision_ESP32");
    Serial.println("Bluetooth aktif: BlindVision_ESP32");

    sensor.setTimeout(500);
    if (!sensor.init()) {
        Serial.println("HATA: VL53L1X bulunamadi! Simule moda geciliyor.");
    } else {
        sensor.setDistanceMode(VL53L1X::Long);
        sensor.setMeasurementTimingBudget(50000);
        sensor.startContinuous(50);
    }
}

void loop() {
    if (millis() - lastSend >= INTERVAL_MS) {
        lastSend = millis();
        sensor.read();

        if (!sensor.timeoutOccurred() && sensor.ranging_data.range_status == 0) {
            int rawCm = sensor.ranging_data.range_mm / 10;
            int smoothCm = getFilteredDistance(rawCm);

            if (SerialBT.hasClient()) {
                SerialBT.printf("DIST:%d\n", smoothCm);
            }
            Serial.printf("DIST:%d\n", smoothCm);
        }
    }
}
