package com.example.emotionclassifierapp

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.util.Log
import android.widget.ImageButton
import android.widget.ProgressBar
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.*

class MainActivity : AppCompatActivity() {

    private lateinit var resultText: TextView
    private lateinit var micButton: ImageButton
    private lateinit var progressBar: ProgressBar
    private lateinit var interpreter: Interpreter
    private lateinit var labels: List<String>
    private var isRecording = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        resultText = findViewById(R.id.resultText)
        micButton = findViewById(R.id.micButton)
        progressBar = findViewById(R.id.progressBar)

        labels = assets.open("labels.txt").bufferedReader().readLines()
        interpreter = Interpreter(loadModelFile("emotion_model.tflite"))

        micButton.setOnClickListener {
            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
                requestPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
            } else {
                toggleRecording()
            }
        }
    }

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) toggleRecording()
        }

    private fun toggleRecording() {
        isRecording = !isRecording
        progressBar.visibility = if (isRecording) ProgressBar.VISIBLE else ProgressBar.GONE
        if (isRecording) {
            resultText.text = "Listening..."
            startLiveClassification()
        } else {
            resultText.text = "Stopped"
        }
    }

    private fun startLiveClassification() {
        val sampleRate = 16000
        val bufferSize = AudioRecord.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )

        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            Log.e("Permission", "RECORD_AUDIO permission not granted")
            return
        }

        val audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufferSize
        )

        val buffer = ShortArray(bufferSize)
        val floatBuffer = FloatArray(bufferSize)

        audioRecord.startRecording()

        lifecycleScope.launch(Dispatchers.IO) {
            while (isRecording) {
                val read = audioRecord.read(buffer, 0, buffer.size)
                for (i in 0 until read) {
                    floatBuffer[i] = buffer[i] / 32768f
                }
                val melInput = extractMelSpectrogramFromFloatArray(floatBuffer.copyOf(read), sampleRate)
                val inputBuffer = ByteBuffer.allocateDirect(4 * 100 * 40).order(ByteOrder.nativeOrder())
                melInput.forEach { inputBuffer.putFloat(it) }
                val output = Array(1) { FloatArray(4) }
                interpreter.run(inputBuffer, output)
                val predictedIndex = output[0].indices.maxByOrNull { output[0][it] } ?: -1
                val label = labels[predictedIndex]
                withContext(Dispatchers.Main) {
                    resultText.text = "Detected: $label (${(output[0][predictedIndex] * 100).toInt()}%)"
                }
                delay(2000)
            }
            audioRecord.stop()
            audioRecord.release()
        }
    }

    private fun loadModelFile(fileName: String): MappedByteBuffer {
        val fileDescriptor = assets.openFd(fileName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun extractMelSpectrogramFromFloatArray(samples: FloatArray, sampleRate: Int): FloatArray {
        val frameSize = 512
        val hopSize = 256
        val numMelBands = 40
        val fftSize = frameSize / 2 + 1

        val frames = samples.asList().windowed(frameSize, hopSize, partialWindows = true)
        val output = mutableListOf<FloatArray>()

        for (frame in frames.take(100)) {
            val windowed = FloatArray(frameSize) { i ->
                val sample = if (i < frame.size) frame[i] else 0f
                val hann = 0.5f * (1 - cos(2 * Math.PI * i / (frameSize - 1))).toFloat()
                sample * hann
            }
            val magnitude = FloatArray(fftSize)
            for (k in 0 until fftSize) {
                var sumReal = 0f
                var sumImag = 0f
                for (n in 0 until frameSize) {
                    val angle = 2 * Math.PI * k * n / frameSize
                    sumReal += windowed[n] * cos(angle).toFloat()
                    sumImag -= windowed[n] * sin(angle).toFloat()
                }
                magnitude[k] = sumReal * sumReal + sumImag * sumImag
            }
            val mel = melFilterBank(magnitude, sampleRate, fftSize, numMelBands)
            val logMel = mel.map { log10(max(it, 1e-10f)) }.toFloatArray()
            output.add(logMel)
        }

        val padded = List(100) { i ->
            if (i < output.size) output[i] else FloatArray(numMelBands) { 0f }
        }
        val flat = padded.flatMap { it.toList() }
        val mean = flat.average().toFloat()
        val std = sqrt(flat.map { (it - mean).pow(2) }.average().toFloat()).takeIf { it != 0f } ?: 1f
        val normalized = flat.map { (it - mean) / std }
        return FloatArray(100 * numMelBands) { i -> normalized[i] }
    }

    private fun melFilterBank(powerSpectrum: FloatArray, sampleRate: Int, fftSize: Int, numMelBands: Int): FloatArray {
        val melMin = 0f
        val melMax = 2595 * log10(1 + sampleRate / 2f / 700)
        val melPoints = FloatArray(numMelBands + 2) { i ->
            val mel = melMin + i * (melMax - melMin) / (numMelBands + 1)
            (700 * ((10.0.pow(mel / 2595.0)) - 1)).toFloat()
        }
        val bin = melPoints.map { (it / (sampleRate / 2) * (fftSize - 1)).roundToInt() }

        val filterBank = Array(numMelBands) { FloatArray(fftSize) { 0f } }
        for (m in 1 until bin.size - 1) {
            val f_m_minus = bin[m - 1]
            val f_m = bin[m]
            val f_m_plus = bin[m + 1]

            for (k in f_m_minus until f_m) {
                filterBank[m - 1][k] = (k - f_m_minus).toFloat() / (f_m - f_m_minus)
            }
            for (k in f_m until f_m_plus) {
                filterBank[m - 1][k] = (f_m_plus - k).toFloat() / (f_m_plus - f_m)
            }
        }

        return FloatArray(numMelBands) { m ->
            var sum = 0f
            for (k in 0 until fftSize) {
                sum += powerSpectrum[k] * filterBank[m][k]
            }
            sum
        }
    }
}
