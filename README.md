# Autonomous-Driving-Simulation-Using-NVIDIA-Based-Behavioral-Cloning
📌 Project OverviewThis project implements an end-to-end self-driving car simulation using Behavioral Cloning in the Udacity Self-Driving Car Simulator. The model is trained using Convolutional Neural Networks (CNNs) to predict steering angles based on camera input, allowing the vehicle to drive autonomously.
The approach mimics real-world ADAS (Advanced Driver Assistance Systems) and Autonomous Driving (AD) methodologies, demonstrating essential computer vision, deep learning, and real-time inference techniques.
📢 Key Features✅ Data Collection & Preprocessing: Extracted driving data from three camera views (left, center, right) to train a robust model.
✅ Deep Learning for Steering Prediction: Implemented a CNN-based on NVIDIA’s End-to-End Learning Model for real-time inference.
✅ Regression-Based Steering Control: Designed a continuous steering angle prediction system, ensuring smooth driving.
✅ Model Deployment & Real-Time Communication: Integrated Flask and SocketIO for simulator interaction.
✅ Industry-Relevant ADAS/AD Techniques: Applied sensor fusion (multi-camera inputs), deep learning, and computer vision for autonomous driving.
🛠 Technologies UsedDeep Learning Frameworks: TensorFlow, Keras
Computer Vision: OpenCV, PIL (for preprocessing and augmentation)
Neural Network Architecture: NVIDIA’s CNN for Behavioral Cloning
Software & Deployment: Flask, SocketIO, Eventlet (for real-time model inference)
Simulation & Data Collection: Udacity Self-Driving Car Simulator
Data Collection & PreprocessingCaptured images from three perspectives (center, left, right) to train the model.
Converted images to YUV color space (similar to NVIDIA's approach) for improved feature extraction.
Applied data augmentation techniques (flipping, brightness adjustment, cropping) to create a balanced dataset.
Normalized pixel values for optimal CNN training.
📈 Model ArchitectureThe CNN used is based on NVIDIA’s End-to-End Learning Model, which consists of:
Convolutional Layers (to extract spatial features from images)
Fully Connected Layers (to learn steering control from extracted features)
Dropout Layers (to prevent overfitting)
Activation Functions (ReLU for non-linearity)
📌 Loss Function: Mean Squared Error (MSE) for continuous steering angle prediction.
📌 Optimizer: Adam Optimizer for efficient convergence.
🚀 Training the ModelTo train the model, run:
Options:
--epochs: Number of training epochs (default: 10)
--batch_size: Batch size for training (default: 32)
After training, a model.h5 file is generated, which can be used for inference.
Running the SimulationTo test the trained model in the simulator, execute:
python drive.py model.h5This script connects to the simulator using Flask & SocketIO, allowing real-time steering prediction.
The model receives camera feed, processes it through the CNN, and outputs the steering angle.
