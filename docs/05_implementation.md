5. Implementation Details
5.1. Machine Learning Models
Both agents are powered by GRU (Gated Recurrent Unit) Autoencoder models. This architecture is highly effective at learning patterns in sequential data. The models are trained to reconstruct "normal" user behavior with low error. When presented with an anomalous pattern, the reconstruction error spikes, signaling an anomaly. The final models are converted to the .tflite format with quantization to ensure high performance and a small footprint on the device.

5.2. Android Application
The application is built natively in Kotlin. The core logic is separated into classes for each agent and the coordinator, with MainActivity.kt serving as the UI and event handler. The TFLite Interpreter is initialized with a FlexDelegate, which is critical for enabling the advanced operations required by the GRU models.
