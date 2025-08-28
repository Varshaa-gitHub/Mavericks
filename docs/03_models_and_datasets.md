3. Models and Datasets
Models Used
This project utilizes open-weight models as its foundation. As no pre-trained models for this specific behavioral analysis task were available, we developed our own.

Models Published
The two GRU autoencoder models developed for this solution will be published on Hugging Face under an Apache 2.0 license.

Movement Anomaly Model ,Typing Anomaly Model: https://huggingface.co/Varshaa7-M/sensor_anomaly_model/tree/main

Datasets Used
The models were trained on proprietary datasets collected specifically for this project, as no public datasets perfectly matched the required data structure (sequences of raw accelerometer data and keystroke latencies for individual users).

Datasets Published
The synthetic and proprietary datasets created for this project will be anonymized and published on Hugging Face under the Open Data Commons Attribution License.

Movement Behavior Dataset: https://huggingface.co/datasets/Varshaa7-M/accelerometer/tree/main

Typing Rhythm Dataset: https://huggingface.co/datasets/Varshaa7-M/typingRhythm/tree/main
