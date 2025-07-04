# Emotion Classifier Android App

This is a real-time **speech emotion recognition Android application** that classifies live or recorded audio into one of four emotions: `angry`, `happy`, `neutral`, or `sad`.

Built using:
- TensorFlow Lite (TFLite)
- Kotlin + Android SDK
- Real-time audio recording
- Mel spectrogram feature extraction (on-device)

---

## Features
-  **Live audio classification** using microphone input
- **Mel spectrogram preprocessing** without external libraries
- Offline model inference using TensorFlow Lite
-  Lightweight and mobile-optimized model
-  Simple UI with mic icon and real-time detection

---

## Project Structure
```
EmotionClassifierApp/
├── app/
│   ├── src/
│   │   └── main/
│   │       ├── java/com/example/emotionclassifierapp/MainActivity.kt
│   │       ├── assets/
│   │       │   ├── emotion_model.tflite
│   │       │   └── labels.txt
│   │       └── res/
│   │           └── layout/activity_main.xml
│   └── build.gradle
├── build.gradle
└── README.md
```

---

## Getting Started

### 1. Clone the repo:
```bash
git clone https://github.com/your-username/emotion-classifier-android.git
cd EmotionClassifierApp
```

### 2. Open in Android Studio
- Open the project folder
- Let Gradle sync
- Make sure you have an emulator or real device connected

### 3. Run the App
- Build > Build APK(s) OR click the green Run ▶️ button
- If using a real device:
  - Enable **Developer Options > USB Debugging**
  - Use **ADB install** or manually transfer the APK

### 4. Permissions
Ensure microphone permission is granted. The app will prompt at first launch.

---

## Model Details
- Trained on RAVDESS dataset with 4 target classes
- Input: `(100, 40)` log-Mel spectrogram
- Quantized for fast inference on mobile

---

## How It Works
1. Records audio using `AudioRecord`
2. Splits audio into frames (512 samples per frame)
3. Applies Hann window and FFT
4. Builds 40-band Mel filterbank
5. Computes log-Mel spectrogram → normalizes → reshapes to (100x40)
6. Runs through `.tflite` model for prediction


## License
MIT License

---
