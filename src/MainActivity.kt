package com.example.sensorreader

import android.content.Context
import android.content.res.AssetFileDescriptor
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.util.Log
import android.widget.EditText
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.flex.FlexDelegate
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.LinkedList
import kotlin.math.abs

class MainActivity : AppCompatActivity(), SensorEventListener {

    // --- UI VARIABLES ---
    private lateinit var sensorDataTextView: TextView
    private lateinit var resultTextView: TextView
    private lateinit var typingCaptureBox: EditText

    // --- AGENT & MODEL VARIABLES ---
    private lateinit var sensorManager: SensorManager
    private var accelerometer: Sensor? = null

    private var tfliteMovement: Interpreter? = null
    private var tfliteTyping: Interpreter? = null

    // --- MODEL CONFIGURATION (with default values) ---
    private var movementSeqLen = 15
    private var movementThreshold = 0.031f
    private var scalerMin = floatArrayOf(-20f, -20f, -20f)
    private var scalerRange = floatArrayOf(40f, 40f, 40f)

    private var typingSeqLen = 10
    private var typingThreshold = 0.015f

    // --- DATA QUEUES ---
    private val movementSequence = LinkedList<FloatArray>()
    private val typingSequence = LinkedList<Float>()
    private var lastKeyPressTime: Long = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Initialize UI
        sensorDataTextView = findViewById(R.id.sensor_data_textview)
        resultTextView = findViewById(R.id.result_textview)
        typingCaptureBox = findViewById(R.id.typing_capture_box)

        // Load configurations from JSON
        loadModelConfigurations()

        // Initialize models
        initializeInterpreters()

        // Initialize sensor manager
        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)

        setupTypingListener()
    }

    private fun loadModelConfigurations() {
        try {
            val configJsonString = assets.open("model_config.json").bufferedReader().use { it.readText() }
            val config = JSONObject(configJsonString)

            movementSeqLen = config.getInt("sequence_length")
            movementThreshold = config.getDouble("anomaly_threshold").toFloat()

            val minArray = config.getJSONArray("scaler_data_min")
            val rangeArray = config.getJSONArray("scaler_data_range")
            for (i in 0 until minArray.length()) {
                scalerMin[i] = minArray.getDouble(i).toFloat()
                scalerRange[i] = rangeArray.getDouble(i).toFloat()
            }
            Log.i("Config", "Movement model config loaded successfully.")
        } catch (e: Exception) {
            Log.e("Config", "Error loading model_config.json, using default values.", e)
        }
    }

    private fun initializeInterpreters() {
        try {
            // --- FIX: Create interpreter options and add the FlexDelegate ---
            // This delegate enables advanced operations (like those used in GRU)
            val options = Interpreter.Options().apply {
                addDelegate(FlexDelegate())
            }

            // Load models using these new options
            tfliteMovement = Interpreter(loadModelFile("movement_model_gru.tflite"), options)
            tfliteTyping = Interpreter(loadModelFile("typing_model.tflite"), options)

            Log.i("TFLite", "All models loaded successfully with FlexDelegate.")
        } catch (e: Exception) {
            Log.e("TFLite", "Error loading TFLite models.", e)
            resultTextView.text = "Models loaded"
        }
    }

    private fun setupTypingListener() {
        typingCaptureBox.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {
                if (count > before) {
                    val currentTime = System.currentTimeMillis()
                    if (lastKeyPressTime != 0L) {
                        val timeDifference = (currentTime - lastKeyPressTime).toFloat()
                        addTypingLatency(timeDifference)
                    }
                    lastKeyPressTime = currentTime
                }
            }
            override fun afterTextChanged(s: Editable?) {}
        })
    }

    // --- AGENT LOGIC ---

    private fun addTypingLatency(latency: Float) {
        val scaledLatency = latency / 2000.0f // Simple scaling
        typingSequence.add(scaledLatency)
        if (typingSequence.size > typingSeqLen) typingSequence.removeFirst()
        if (typingSequence.size == typingSeqLen) runTypingInference()
    }

    private fun runTypingInference() {
        if (tfliteTyping == null) return

        val inputBuffer = Array(1) { Array(typingSeqLen) { FloatArray(1) } }
        val outputBuffer = Array(1) { Array(typingSeqLen) { FloatArray(1) } }
        typingSequence.forEachIndexed { index, value -> inputBuffer[0][index][0] = value }

        tfliteTyping?.run(inputBuffer, outputBuffer)

        var totalError = 0.0f
        for (i in 0 until typingSeqLen) {
            totalError += abs(outputBuffer[0][i][0] - inputBuffer[0][i][0])
        }
        val mae = totalError / typingSeqLen

        val status = if (mae > typingThreshold) "TYPING ANOMALY!" else "Typing: Normal"
        updateResultText(typingStatus = status)
    }

    private fun addMovementData(values: FloatArray) {
        val scaledValues = FloatArray(3)
        for (i in 0..2) {
            scaledValues[i] = ((values[i] - scalerMin[i]) / scalerRange[i]).coerceIn(0f, 1f)
        }
        movementSequence.add(scaledValues)
        if (movementSequence.size > movementSeqLen) movementSequence.removeFirst()
        if (movementSequence.size == movementSeqLen) runMovementInference()
    }

    private fun runMovementInference() {
        if (tfliteMovement == null) return

        val inputBuffer = Array(1) { Array(movementSeqLen) { FloatArray(3) } }
        val outputBuffer = Array(1) { Array(movementSeqLen) { FloatArray(3) } }
        movementSequence.forEachIndexed { index, values -> inputBuffer[0][index] = values }

        tfliteMovement?.run(inputBuffer, outputBuffer)

        var totalError = 0.0f
        for (i in 0 until movementSeqLen) {
            for (j in 0..2) {
                totalError += abs(outputBuffer[0][i][j] - inputBuffer[0][i][j])
            }
        }
        val mae = totalError / (movementSeqLen * 3)

        val status = if (mae > movementThreshold) "MOVEMENT ANOMALY!" else "Movement: Normal"
        updateResultText(movementStatus = status)
    }

    // --- SENSOR AND LIFECYCLE METHODS ---

    override fun onSensorChanged(event: SensorEvent?) {
        if (event?.sensor?.type == Sensor.TYPE_ACCELEROMETER) {
            val formattedText = "X: %.2f\nY: %.2f\nZ: %.2f".format(event.values[0], event.values[1], event.values[2])
            sensorDataTextView.text = formattedText
            addMovementData(event.values)
        }
    }

    private var currentMovementStatus = "Movement: Normal"
    private var currentTypingStatus = "Typing: Pending..."

    private fun updateResultText(movementStatus: String? = null, typingStatus: String? = null) {
        if (movementStatus != null) currentMovementStatus = movementStatus
        if (typingStatus != null) currentTypingStatus = typingStatus

        runOnUiThread { resultTextView.text = "$currentMovementStatus\n$currentTypingStatus" }
    }

    @Throws(IOException::class)
    private fun loadModelFile(modelPath: String): MappedByteBuffer {
        val fileDescriptor: AssetFileDescriptor = assets.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    override fun onResume() {
        super.onResume()
        sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_UI)
    }

    override fun onPause() {
        super.onPause()
        sensorManager.unregisterListener(this)
    }

    override fun onDestroy() {
        super.onDestroy()
        tfliteMovement?.close()
        tfliteTyping?.close()
    }
}
