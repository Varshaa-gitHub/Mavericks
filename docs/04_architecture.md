4. Technical Architecture
The system is designed as a modular, multi-agent system where each component has a distinct responsibility.

4.1. Agent Breakdown
Movement Agent (MovementAgent.kt): Analyzes the physical movement of the device using the accelerometer and a dedicated movement_model_gru.tflite.

Typing Agent (TypingAgent.kt): Analyzes the user's typing rhythm using a TextWatcher and a typing_model.tflite.

Coordinator Agent (Coordinator.kt): Acts as the central decision-making brain, receiving anomaly signals from all agents and applying risk assessment logic.

4.2. Data and Logic Flow
Data Sensing: The MainActivity captures raw data from device sensors.

Data Delegation: The raw data is passed to the Coordinator.

Agent Processing: The Coordinator forwards the data to the appropriate agent.

On-Device Inference: Each agent maintains a sequence of recent data. When a sequence is full, it is fed into the agent's .tflite model for inference.

Anomaly Reporting: Each agent calculates the model's reconstruction error and compares it to a pre-defined threshold to determine if an anomaly has occurred.

Risk Assessment & Action: The Coordinator collects the anomaly statuses and, if the context is sensitive, triggers a security response.
