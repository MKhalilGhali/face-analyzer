import cv2
import numpy as np

class FaceDetector:
    def __init__(self):
        """
        Initialize face detector using OpenCV's Haar cascade
        More reliable and doesn't require external model files
        """
        try:
            # Load Haar cascade for face detection
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            print("Face detector initialized successfully!")
        except Exception as e:
            print(f"Error initializing face detector: {e}")
            raise
    
    def detect_faces(self, frame):
        """
        Detect faces in the frame using Haar cascade
        Returns list of face coordinates (x, y, w, h)
        """
        try:
            # Convert to grayscale for detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            return faces
            
        except Exception as e:
            print(f"Face detection error: {e}")
            return []