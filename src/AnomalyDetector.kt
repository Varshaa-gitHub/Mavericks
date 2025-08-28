package com.example.sensorreader
import android.content.Context
import com.google.gson.Gson
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.sqrt

// Data class for a single raw sensor reading
data class SensorData(val x: Float, val y: Float, val z: Float)

// Data class for the final prediction result
data class AnomalyResult(val isAnomaly: Boolean, val errorScore: Float)

class AnomalyDetector(private val context: Context) {

    // --- Configuration and Model Properties ---
    private var interpreter: Interpreter? = null
    private var sequenceLength: Int = 0
    private var windowSize: Int = 0
    private var numFeatures: Int = 0
    private var anomalyThreshold: Float = 0.0f
    private lateinit var scalerMin: FloatArray
    private lateinit var scalerScale: FloatArray

    // --- Data Buffer ---
    // This buffer holds the raw sensor data needed to generate one full sequence for the model.
    private val rawDataBuffer = mutableListOf<SensorData>()
    private var requiredBufferSize = 0

    var isInitialized = false
        private set

    // --- Public Methods ---

    /**
     * Loads the TFLite model and configuration from assets. Must be called once before use.
     */
    fun initialize() {
        try {
            loadConfig() // Load parameters from JSON

            val modelBuffer = loadModelFile()
            val options = Interpreter.Options()
            interpreter = Interpreter(modelBuffer, options)

            // The total number of raw readings needed is for the first window plus the subsequent sequence steps.
            // Example: seq_len=20, win_size=10 -> need 10 + (20-1) = 29 raw readings.
            requiredBufferSize = windowSize + (sequenceLength - 1)

            isInitialized = true
        } catch (e: Exception) {
            isInitialized = false
            // In a real app, log this error for debugging
            e.printStackTrace()
        }
    }

    /**
     * Main entry point for the detector. Adds a new reading and runs prediction if the buffer is full.
     * @param reading The new SensorData object from the device's sensor.
     * @return AnomalyResult if a prediction was made, otherwise null.
     */
    fun addReadingAndPredict(reading: SensorData): AnomalyResult? {
        if (!isInitialized) return null

        rawDataBuffer.add(reading)
        // Trim the buffer from the front to maintain its required size
        if (rawDataBuffer.size > requiredBufferSize) {
            rawDataBuffer.removeAt(0)
        }

        // Only perform a prediction if we have exactly the right amount of data
        if (rawDataBuffer.size == requiredBufferSize) {
            return predictOnBuffer(rawDataBuffer)
        }

        return null // Not enough data yet
    }

    // --- Private Helper Methods ---

    /**
     * Performs the full prediction pipeline: feature engineering, scaling, and inference.
     */
    private fun predictOnBuffer(readings: List<SensorData>): AnomalyResult? {
        // 1. Create the sequence of feature sets from the raw data buffer
        val featureSequence = Array(sequenceLength) { FloatArray(numFeatures) }
        for (i in 0 until sequenceLength) {
            // Get the slice of raw data for the current window
            val window = readings.subList(i, i + windowSize)
            featureSequence[i] = calculateFeatures(window)
        }

        // 2. Scale the features and prepare the input buffer for the TFLite model
        val inputBuffer = prepareInputBuffer(featureSequence)

        // 3. Prepare the output buffer to receive the reconstructed features
        val outputBuffer = ByteBuffer.allocateDirect(1 * sequenceLength * numFeatures * 4).apply {
            order(ByteOrder.nativeOrder())
        }

        // 4. Run inference using the TFLite interpreter
        interpreter?.run(inputBuffer, outputBuffer)

        // 5. Calculate the Mean Absolute Error between the input features and the reconstructed features
        outputBuffer.rewind()
        var totalAbsoluteError = 0.0f
        for (i in 0 until sequenceLength) {
            val scaledInputFeatures = scale(featureSequence[i])

            for (j in 0 until numFeatures) {
                val reconstructedFeature = outputBuffer.float
                totalAbsoluteError += kotlin.math.abs(scaledInputFeatures[j] - reconstructedFeature)
            }
        }
        val mae = totalAbsoluteError / (sequenceLength * numFeatures)

        // 6. Compare the error score to the threshold to determine if it's an anomaly
        return AnomalyResult(isAnomaly = mae > anomalyThreshold, errorScore = mae)
    }

    /**
     * Calculates statistical features over a window of raw sensor data.
     * THIS LOGIC MUST EXACTLY MATCH THE FEATURE ENGINEERING IN THE PYTHON SCRIPT.
     */
    private fun calculateFeatures(window: List<SensorData>): FloatArray {
        val size = window.size.toFloat()
        var sumX = 0.0f; var sumY = 0.0f; var sumZ = 0.0f

        window.forEach {
            sumX += it.x; sumY += it.y; sumZ += it.z
        }
        val meanX = sumX / size
        val meanY = sumY / size
        val meanZ = sumZ / size

        var sumSqDiffX = 0.0f; var sumSqDiffY = 0.0f; var sumSqDiffZ = 0.0f
        window.forEach {
            sumSqDiffX += (it.x - meanX) * (it.x - meanX)
            sumSqDiffY += (it.y - meanY) * (it.y - meanY)
            sumSqDiffZ += (it.z - meanZ) * (it.z - meanZ)
        }
        // Use (size - 1) for sample standard deviation, matching pandas .std()
        val stdX = if (size > 1) sqrt(sumSqDiffX / (size - 1)) else 0.0f
        val stdY = if (size > 1) sqrt(sumSqDiffY / (size - 1)) else 0.0f
        val stdZ = if (size > 1) sqrt(sumSqDiffZ / (size - 1)) else 0.0f

        // The order must be consistent with the Python script
        return floatArrayOf(meanX, meanY, meanZ, stdX, stdY, stdZ)
    }

    /**
     * Scales a feature set using the loaded scaler parameters.
     * Formula: scaled = (value - min) * scale
     */
    private fun scale(features: FloatArray): FloatArray {
        val scaled = FloatArray(numFeatures)
        for (i in 0 until numFeatures) {
            scaled[i] = (features[i] - scalerMin[i]) * scalerScale[i]
        }
        return scaled
    }

    private fun prepareInputBuffer(featureSequence: Array<FloatArray>): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(1 * sequenceLength * numFeatures * 4).apply {
            order(ByteOrder.nativeOrder())
        }
        featureSequence.forEach { featureSet ->
            val scaledFeatures = scale(featureSet)
            scaledFeatures.forEach { buffer.putFloat(it) }
        }
        return buffer.rewind() as ByteBuffer
    }

    private fun loadModelFile(): ByteBuffer {
        val assetFileDescriptor = context.assets.openFd("sensor_anomaly_model.tflite")
        FileInputStream(assetFileDescriptor.fileDescriptor).use { fis ->
            val fileChannel = fis.channel
            val startOffset = assetFileDescriptor.startOffset
            val declaredLength = assetFileDescriptor.declaredLength
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        }
    }

    private fun loadConfig() {
        val jsonString = context.assets.open("model_config.json").bufferedReader().use { it.readText() }
        val config = Gson().fromJson(jsonString, ModelConfig::class.java)
        sequenceLength = config.sequence_length
        windowSize = config.window_size
        numFeatures = config.num_features
        anomalyThreshold = config.anomaly_threshold
        scalerMin = config.scaler_min.toFloatArray()
        scalerScale = config.scaler_scale.toFloatArray()
    }

    // Data class to match the structure of the generated model_config.json
    private data class ModelConfig(
        val sequence_length: Int,
        val window_size: Int,
        val num_features: Int,
        val anomaly_threshold: Float,
        val scaler_min: List<Float>,
        val scaler_scale: List<Float>
    )
}
