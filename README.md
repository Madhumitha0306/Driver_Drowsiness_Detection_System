Driver Drowsiness Detection System

A real-time AI-based driver drowsiness detection system that monitors the driver's eyes using Computer Vision and alerts the driver when signs of sleepiness or fatigue are detected. This project helps improve road safety by reducing accidents caused by drowsy driving.

Project Overview

Driver fatigue is one of the major causes of road accidents worldwide. This project uses Python, OpenCV, and Machine Learning / Deep Learning techniques to continuously monitor the driver's facial features and detect drowsiness in real time.

The system captures live video through a webcam, detects the driver's face and eyes, and determines whether the eyes remain closed for a prolonged period. If drowsiness is detected, an alarm alert is triggered to wake the driver.

Research studies also highlight that driver drowsiness detection systems significantly help in reducing road accidents caused by fatigue.

✨ Features

Real-time face and eye detection
Detects prolonged eye closure
Audio alert/alarm system
Live webcam monitoring
AI-based fatigue detection
Easy to run and lightweight
Improves driver safety

🛠️ Technologies Used

1. Python
2. OpenCV
3. NumPy
4.Dlib / Haar Cascade
5. Webcam Video Streaming

⚙️ How It Works

The webcam captures live video frames.
The system detects the driver's face.
Eyes are extracted from the detected face region.
The model predicts whether the eyes are open or closed.
If the eyes remain closed for a certain number of frames, the system identifies drowsiness.
An alarm sound is triggered to alert the driver.


🧠 Detection Method

The project uses Computer Vision techniques such as:

Face Detection
Eye Detection
Eye Aspect Ratio (EAR)
CNN-based Eye Classification (if implemented)

The model continuously analyzes eye movements and blinking patterns to determine driver alertness.

📸 Output

Detects face and eyes in real time
Displays drowsiness warning on screen
Plays alarm when the driver appears sleepy

🚀 Installation

1️⃣ Clone the Repository
git clone https://github.com/Madhumitha0306/Driver_Drowsiness_Detection_System.git

2️⃣ Navigate to the Project Folder
cd Driver_Drowsiness_Detection_System

3️⃣ Install Required Packages
pip install -r requirements.txt
▶️ Run the Project

python main.py

📋 Requirements

Example libraries used:

opencv-python
numpy
dlib
imutils
scipy
tensorflow
keras
pygame

🎯 Applications

Smart vehicle safety systems
Driver monitoring systems
Accident prevention systems
AI-powered transportation safety

📈 Future Enhancements

Mobile application integration
IoT-based vehicle monitoring
Yawning detection
Head pose estimation
Night vision support
Cloud monitoring dashboard

🧪 Dataset

The project can be trained using eye-state datasets containing:

Open eyes
Closed eyes
Fatigue detection images

Example datasets commonly used in drowsiness detection research include eye-state image datasets and facial landmark datasets.

🔔 Alert System

When drowsiness is detected:

Warning message appears on the screen
Alarm sound is played
Driver is alerted instantly
📚 References
Driver drowsiness detection research papers
OpenCV documentation
CNN-based fatigue detection systems

🤝 Contributing

Contributions are welcome!

If you'd like to improve this project:

Fork the repository
Create a new branch
Commit your changes
Submit a pull request
