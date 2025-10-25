import cv2
import numpy as np
from utils import preprocess_face
import time

class EnhancedAgeRecognizer:
    def __init__(self, use_ai=True):
        """
        Initialize enhanced age recognition with AI models
        """
        self.use_ai = use_ai
        self.ai_initialized = False
        self.cache = {}
        self.cache_timeout = 0.5  # Cache predictions for 0.5 seconds
        
        if use_ai:
            self._initialize_ai_model()
        
        print(f"✅ Enhanced Age Recognizer ready! (AI: {'ON' if self.ai_initialized else 'OFF'})")
    
    def _initialize_ai_model(self):
        """
        Initialize AI model with multiple fallback options
        """
        # Try DeepFace first (most accurate)
        try:
            from deepface import DeepFace
            self.DeepFace = DeepFace
            self.ai_initialized = True
            self.model_type = 'deepface'
            print("🤖 DeepFace AI model loaded successfully!")
            return
        except Exception as e:
            print(f"⚠️ DeepFace not available: {e}")
        
        # Try loading a custom age model (OpenCV DNN)
        try:
            self._load_opencv_age_model()
            return
        except Exception as e:
            print(f"⚠️ OpenCV age model not available: {e}")
        
        print("⚠️ No AI models available, using enhanced heuristics only")
        self.ai_initialized = False
    
    def _load_opencv_age_model(self):
        """
        Load OpenCV DNN age estimation model
        You can download pre-trained models from:
        https://github.com/GilLevi/AgeGenderDeepLearning
        """
        try:
            # Age model files (if available)
            age_proto = "age_deploy.prototxt"
            age_model = "age_net.caffemodel"
            
            self.age_net = cv2.dnn.readNet(age_model, age_proto)
            self.age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', 
                            '(25-32)', '(38-43)', '(48-53)', '(60-100)']
            self.ai_initialized = True
            self.model_type = 'opencv_dnn'
            print("🤖 OpenCV DNN age model loaded successfully!")
        except Exception as e:
            raise Exception(f"Could not load OpenCV age model: {e}")
    
    def predict_age(self, face_roi, face_rect=None):
        """
        Enhanced age prediction with AI + heuristics hybrid approach
        Returns: age_range (str), exact_age (int), confidence (float)
        """
        # Check cache
        cache_key = self._get_cache_key(face_rect)
        if cache_key in self.cache:
            cached_time, cached_result = self.cache[cache_key]
            if time.time() - cached_time < self.cache_timeout:
                return cached_result
        
        # Try AI prediction first
        if self.ai_initialized:
            ai_result = self._predict_with_ai(face_roi)
            if ai_result[0] != "Unknown":
                self.cache[cache_key] = (time.time(), ai_result)
                return ai_result
        
        # Fallback to enhanced heuristics
        heuristic_result = self._predict_with_heuristics(face_roi, face_rect)
        self.cache[cache_key] = (time.time(), heuristic_result)
        return heuristic_result
    
    def _predict_with_ai(self, face_roi):
        """
        Predict age using AI models
        """
        try:
            if self.model_type == 'deepface':
                return self._predict_deepface(face_roi)
            elif self.model_type == 'opencv_dnn':
                return self._predict_opencv_dnn(face_roi)
        except Exception as e:
            print(f"AI prediction error: {e}")
        
        return "Unknown", 0, 0.0
    
    def _predict_deepface(self, face_roi):
        """
        Predict age using DeepFace
        """
        try:
            # Preprocess face
            processed_face = preprocess_face(face_roi)
            if processed_face is None:
                return "Unknown", 0, 0.0
            
            # Convert grayscale to BGR if needed
            if len(processed_face.shape) == 2:
                processed_face = cv2.cvtColor(processed_face, cv2.COLOR_GRAY2BGR)
            
            # Use DeepFace to analyze age
            analysis = self.DeepFace.analyze(
                processed_face, 
                actions=['age'],
                enforce_detection=False,
                silent=True,
                detector_backend='skip'  # Skip detection, we already have the face
            )
            
            if isinstance(analysis, list):
                analysis = analysis[0]
            
            age = int(analysis['age'])
            
            # Apply realistic age adjustment based on common model biases
            age = self._adjust_age_prediction(age, face_roi)
            
            age_range = self._get_age_range(age)
            confidence = 0.85  # DeepFace is generally reliable
            
            return age_range, age, confidence
            
        except Exception as e:
            print(f"DeepFace prediction error: {e}")
            return "Unknown", 0, 0.0
    
    def _predict_opencv_dnn(self, face_roi):
        """
        Predict age using OpenCV DNN model
        """
        try:
            # Preprocess for DNN
            blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227),
                                        (78.4263377603, 87.7689143744, 114.895847746),
                                        swapRB=False)
            
            self.age_net.setInput(blob)
            age_preds = self.age_net.forward()
            age_idx = age_preds[0].argmax()
            age_range = self.age_list[age_idx]
            
            # Extract approximate age from range
            age = self._extract_age_from_range(age_range)
            confidence = float(age_preds[0][age_idx])
            
            return age_range, age, confidence
            
        except Exception as e:
            print(f"OpenCV DNN prediction error: {e}")
            return "Unknown", 0, 0.0
    
    def _adjust_age_prediction(self, predicted_age, face_roi):
        """
        Adjust AI predictions based on facial features to improve accuracy
        """
        try:
            # Extract features for adjustment
            h, w = face_roi.shape[:2]
            
            # Calculate skin texture (wrinkles indicator)
            gray = face_roi if len(face_roi.shape) == 2 else cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            texture = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # High texture (wrinkles) might indicate older than predicted
            if texture > 500 and predicted_age < 40:
                predicted_age += 5
            elif texture < 100 and predicted_age > 25:
                predicted_age -= 3
            
            # Face size adjustment (larger faces often older)
            face_area = h * w
            if face_area > 15000 and predicted_age < 35:
                predicted_age += 3
            elif face_area < 5000 and predicted_age > 20:
                predicted_age -= 2
            
            # Ensure reasonable bounds
            predicted_age = max(5, min(90, predicted_age))
            
            return predicted_age
            
        except:
            return predicted_age
    
    def _predict_with_heuristics(self, face_roi, face_rect):
        """
        Enhanced heuristic-based age prediction (fallback)
        """
        try:
            gray = face_roi if len(face_roi.shape) == 2 else cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            age_score = 0
            
            # Face area analysis
            face_area = h * w
            if face_area > 12000:
                age_score += 3
            elif face_area > 8000:
                age_score += 1
            elif face_area < 4000:
                age_score -= 2
            
            # Texture analysis (wrinkles, skin quality)
            texture = cv2.Laplacian(gray, cv2.CV_64F).var()
            if texture > 500:
                age_score += 2  # More wrinkles = older
            elif texture < 100:
                age_score -= 2  # Smooth skin = younger
            
            # Contrast analysis
            contrast = np.std(gray)
            if contrast > 55:
                age_score += 1
            elif contrast < 35:
                age_score -= 1
            
            # Edge density (more edges = more facial features/wrinkles)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (h * w)
            if edge_density > 0.15:
                age_score += 2
            elif edge_density < 0.05:
                age_score -= 1
            
            # Determine age based on score
            if age_score >= 6:
                age_range, age = "60-75", 67
            elif age_score >= 4:
                age_range, age = "50-59", 54
            elif age_score >= 2:
                age_range, age = "40-49", 44
            elif age_score >= 0:
                age_range, age = "30-39", 34
            elif age_score >= -2:
                age_range, age = "25-29", 27
            elif age_score >= -4:
                age_range, age = "18-24", 21
            else:
                age_range, age = "12-17", 15
            
            confidence = 0.6  # Lower confidence for heuristics
            
            return age_range, age, confidence
            
        except Exception as e:
            print(f"Heuristic prediction error: {e}")
            return "Unknown", 0, 0.0
    
    def _get_age_range(self, age):
        """
        Convert exact age to realistic age range
        """
        try:
            age = int(age)
            if age < 3:
                return "0-2"
            elif age < 7:
                return "3-6"
            elif age < 13:
                return "7-12"
            elif age < 18:
                return "13-17"
            elif age < 25:
                return "18-24"
            elif age < 30:
                return "25-29"
            elif age < 40:
                return "30-39"
            elif age < 50:
                return "40-49"
            elif age < 60:
                return "50-59"
            elif age < 75:
                return "60-75"
            else:
                return "75+"
        except:
            return "Unknown"
    
    def _extract_age_from_range(self, age_range):
        """
        Extract approximate age from range string
        """
        try:
            # Extract numbers from range like "(25-32)"
            numbers = [int(s) for s in age_range.replace('(', '').replace(')', '').split('-')]
            return sum(numbers) // len(numbers)
        except:
            return 30
    
    def _get_cache_key(self, face_rect):
        """
        Generate cache key from face rectangle
        """
        if face_rect is None:
            return "default"
        return f"{face_rect[0]}_{face_rect[1]}_{face_rect[2]}_{face_rect[3]}"
    
    def clear_cache(self):
        """
        Clear prediction cache
        """
        self.cache = {}


# Backward compatibility
class AgeRecognizer(EnhancedAgeRecognizer):
    """Alias for backward compatibility"""
    pass