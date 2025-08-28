2. Technical Stack
This project is built entirely on open-source software (OSS) libraries.

Platform: Android (Kotlin) - The native development platform for building the application.

Machine Learning (Training):

TensorFlow/Keras - For building and training the GRU autoencoder models in Python.

Scikit-learn - For data preprocessing, specifically using the MinMaxScaler.

Pandas - For data manipulation and loading CSV files.

Machine Learning (On-Device):

TensorFlow Lite - The core library for running ML models efficiently on mobile devices.

TFLite Flex Delegate - A crucial component that enables the use of advanced TensorFlow operations (like those in GRUs) within the TFLite runtime.
