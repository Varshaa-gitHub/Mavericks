On-Device Multi-Agent Anomaly Detection System
Project for Samsung Ennovatex 2025

This repository contains the source code and documentation for an on-device, multi-agent system designed for real-time, behavior-based anomaly and fraud detection on mobile devices.

🚀 Project Overview
The system establishes a personalized baseline of a user's normal behavior by monitoring on-device signals like movement patterns and typing rhythm. It uses a multi-agent architecture where specialized agents analyze individual data streams. A central coordinator assesses risks based on agent feedback, allowing the system to detect anomalies like bot activity or human spoofing without compromising user privacy by sending data to external servers.

Core Features
Privacy-First: All data collection, model training, and inference happen 100% on-device.

Multi-Agent Architecture: Specialized agents (Movement, Typing) work together for robust detection.

Behavioral Biometrics: Learns the user's unique patterns for movement and typing, going beyond static passwords or PINs.

Real-Time & Efficient: Built with lightweight GRU models (.tflite) for continuous background operation with minimal battery impact.

Context-Aware Security: Designed to trigger security actions only during sensitive operations, preventing unnecessary interruptions.

📚 Technical Documentation
For a complete technical breakdown of the project, please refer to the documents below:

1. Approach and Uniqueness

2. Technical Stack

3. Models and Datasets

4. Technical Architecture

5. Implementation Details

6. Installation Instructions

7. User Guide

8. Salient Features
