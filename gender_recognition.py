import cv2
from utils import preprocess_face

class GenderRecognizer:
    def __init__(self):
        """
        Initialize gender recognition with fallback options
        """
        self.initialized = False
        self._initialize_model()
    
    def _initialize_model(self):
        """
        Initialize DeepFace model with error handling
        """
        try:
            # Import DeepFace only when needed
            from deepface import DeepFace
            self.DeepFace = DeepFace
            self.initialized = True
            print("Gender recognizer initialized successfully!")
        except Exception as e:
            print(f"Warning: Could not initialize DeepFace for gender recognition: {e}")
            print("Gender recognition will be disabled")
            self.initialized = False
    
    def predict_gender(self, face_roi):
        """
        Predict gender from face ROI
        Returns: gender (str), confidence (float)
        """
        if not self.initialized:
            return "Unknown", 0.0
            
        try:
            # Preprocess face
            processed_face = preprocess_face(face_roi)
            if processed_face is None:
                return "Unknown", 0.0
            
            # Use DeepFace to analyze gender
            analysis = self.DeepFace.analyze(
                processed_face, 
                actions=['gender'],
                enforce_detection=False,
                silent=True,
                detector_backend='opencv'
            )
            
            if isinstance(analysis, list):
                analysis = analysis[0]
            
            gender = analysis['gender']
            dominant_gender = max(gender, key=gender.get)
            confidence = gender[dominant_gender] / 100.0
            
            return dominant_gender, confidence
            
        except Exception as e:
            print(f"Gender recognition error: {e}")
            return "Unknown", 0.0