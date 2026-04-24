# Drowsiness & Driver Monitoring System

A Computer Vision Project using Python, MediaPipe, and YOLO

# Project Overview

This project is a self learning to practice. This is a driver monitoring system created to identify signs of drowsiness and risky behavior through computer vision.

The system relies on a webcam to observe the driver’s face and environment, issuing alerts when certain conditions are detected, including:

- Eyes remaining closed for an extended time
- Head leaning or tilting
- Mouth opening (indicating a yawn)
- Use of a phone while driving
- Detection of hands


# Objectives

purpose of this project is to:

Improve road safety through real-time monitoring
Detect driver fatigue and distraction
Provide instant visual alerts
Apply computer vision methods in a scenario



# Features

## Eye Closure Detection
Uses Eye Aspect Ratio (EAR)
Identifies if eyes remain closed for 2 seconds
Displays: 
ALERT !


## Head Tilt Detection
Calculates face angle using facial landmarks
Detects head tilt for 3 seconds
Displays: 
ALERT !


## Mouth Open Detection (Yawning)
Uses Mouth Aspect Ratio (MAR)
Detects mouth open for 3 seconds
Displays: 
ALERT !

## Phone Detection
Uses YOLO (Ultralytics) object detection
Identifies the presence of a cell phone
Triggers an alert if detected for 3 seconds
Displays: 
ALERT !

## Hand Detection
Uses MediaPipe Hands
Detects when a hand is visible
Displays after 3 seconds: 
Hands on the wheel




# Technologies Used
- Python
- OpenCV – video processing
- MediaPipe
- Face Mesh (face landmarks)
- Hands (hand detection)
- Ultralytics YOLOv8 – object detection (phone)
- NumPy / Math – geometric calculations






# Run the project 
''' bash 
python main.py
'''


