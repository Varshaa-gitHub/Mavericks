7. User Guide
The application is designed to run in the background with no direct user interaction required for its security functions.

Training Phase (Implicit): Upon first use, the app begins to silently learn the user's unique movement and typing patterns. For a robust baseline, the user should use their device normally for a few days.

Monitoring Phase: The app runs a persistent background service, continuously monitoring sensor and typing data.

Anomaly Detection: When an anomaly is detected (e.g., a different person is typing), the system logs it.

Security Action: If an anomaly is detected while the user is performing a sensitive action (e.g., opening a banking app), the system will present a biometric prompt (fingerprint/face) to re-verify the user's identity before allowing access.
