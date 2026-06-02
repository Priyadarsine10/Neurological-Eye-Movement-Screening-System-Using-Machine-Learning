**Neurological Eye Movement Screening System**
**Overview**

This project is a real-time eye movement analysis and neurological screening system that combines Computer Vision and Machine Learning techniques. Using a standard webcam and MediaPipe Face Mesh, the system tracks iris movements, extracts gaze-related features, and predicts whether the observed eye movement pattern is normal or abnormal.

The primary objective is to provide a lightweight and cost-effective alternative to traditional eye-tracking systems that require expensive hardware and controlled laboratory environments.

**Problem Statement**

Neurological disorders and behavioral conditions often influence eye movement patterns. Traditional eye-tracking systems used for diagnosis and assessment require specialized equipment, making them expensive and inaccessible for many users.

**Challenges include:**

High cost of commercial eye-tracking devices
Requirement of specialized hardware
Limited accessibility in remote locations
Lack of affordable real-time screening tools

This project addresses these challenges by developing a webcam-based neurological screening system using machine learning and computer vision technologies.

**Dataset Description**

The project utilizes two datasets:

Normal Eye Movement Dataset

Contains gaze-related features representing normal eye movement behavior, including:

Fixation Count
Saccade Count
Mean Velocity
Velocity Standard Deviation
Fixation Ratio
Eye Movement Sequences
Abnormal Eye Movement Dataset

Contains autism-related and abnormal gaze behavior patterns used for classification and model training.

**Data Preprocessing**

The datasets undergo preprocessing steps such as:

Missing value handling
Feature selection
Data cleaning
Feature scaling using StandardScaler
Label encoding
Dataset balancing

These steps improve model performance and prediction reliability.

**Project Workflow**
**1. Data Collection**

Historical gaze feature datasets representing normal and abnormal eye movement behavior are collected and prepared for training.

**2. Data Preprocessing**

The preprocessing pipeline includes:

Cleaning missing values
Feature extraction
Feature scaling
Label assignment
Dataset preparation for machine learning

This ensures consistency and improves model accuracy.

**3. Real-Time Eye Tracking**

The system captures live video through a webcam and performs:

Face detection
Iris landmark detection
Gaze coordinate estimation
Eye movement tracking

MediaPipe Face Mesh is used to extract precise eye landmarks in real time.

**4. Calibration and Gaze Normalization**

Before tracking begins, a calibration stage is performed where users look at the screen center.

The system:

Calculates baseline gaze coordinates
Normalizes gaze positions
Reduces user-specific variations

This improves tracking consistency and prediction performance.

**5. Feature Extraction**

The system extracts important eye movement features from tracked gaze data.

Extracted Features
Fixation Count
Saccade Count
Mean Velocity
Velocity Standard Deviation
Fixation Ratio

These features serve as input for machine learning classification.

**6. Model Training and Classification**

A Logistic Regression classifier is trained using the processed datasets.

The model learns relationships between gaze features and eye movement behavior to classify patterns as:

Normal
Abnormal

The trained model is saved and used for real-time prediction.

**7. Real-Time Prediction**

Once tracking is completed, the system:

Extracts gaze features
Scales input features
Performs prediction
Calculates classification probabilities
Displays the final screening result

The application provides immediate feedback to the user.

Technologies Used
Python
OpenCV
MediaPipe Face Mesh
NumPy
Pandas
Scikit-learn
Streamlit
Joblib
Results

The developed system successfully performs:

Real-time webcam-based eye tracking
Gaze feature extraction
Eye movement classification
Normal/Abnormal prediction
Probability estimation

The lightweight implementation enables efficient neurological screening without requiring specialized hardware.

Applications
Neurological Screening
Behavioral Analysis
Autism Research
Eye Movement Studies
Healthcare Monitoring
AI-Assisted Diagnosis Support
Academic Research
Novelty of the Project

**Key contributions of this work include:**

Webcam-based real-time eye tracking
MediaPipe Face Mesh integration
Calibration-based gaze normalization
Smoothing-based noise reduction
Lightweight Logistic Regression classification
Streamlit-based deployment
Low-cost neurological screening framework
Future Enhancements

**Future developments may include:**

Deep Learning-based classification models
Larger neurological datasets
Multi-disorder detection
Cloud deployment
Mobile application integration
Advanced gaze estimation techniques
Clinical validation studies

**Conclusion**

This project demonstrates how Computer Vision and Machine Learning can be combined to develop an accessible and cost-effective neurological screening system. By leveraging webcam-based eye tracking, feature engineering, and Logistic Regression, the system provides real-time analysis of eye movement behavior and supports AI-assisted healthcare applications.

The framework establishes a strong foundation for future research in intelligent healthcare monitoring and neurological assessment systems.
