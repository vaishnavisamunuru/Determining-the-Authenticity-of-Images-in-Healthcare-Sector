# Determining-the-Authenticity-of-Images-in-Healthcare-Sector
### Aim
* Develop and rigorously evaluate CNN-based detectors of medical image tampering that balance accuracy, robustness, interpretability, and deploy ability within clinical imaging workflows.
### Research Objectives
* Preprocess and assemble a labeled corpus of authentic versus tampered images by combining multiple MedMNIST subsets and applying a diverse library of plausible manipulation techniques that emulate both benign post-processing and malicious edits.
* Train, validate, and compare CNN models (SimpleCNN, ResNet-50, DenseNet-121) using stratified cross-validation, optimizing for accuracy, precision, recall, F1, and ROC-AUC, while monitoring overfitting and calibration.
* Assess cross technique generalization by reporting per manipulation performance across the applied tampering methods, providing an estimate of how well models generalize to different edit types encountered in practice.
* Provide interpretable evidence (Grad-CAM, Integrated Gradients) highlighting suspected tamper regions to support clinical adjudication and mitigate alert fatigue.
