import cv2
import time
import numpy as np
from age_recognition import EnhancedAgeRecognizer
from gender_recognition import GenderRecognizer

class EnhancedFaceAnalyzer:
    def __init__(self, use_ai=True):
        """Enhanced analyzer with AI-powered age detection"""
        print("🚀 Initializing AI-ENHANCED Ultra-Fast System...")
        
        # Load face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Initialize AI models
        self.use_ai = use_ai
        if use_ai:
            try:
                self.age_recognizer = EnhancedAgeRecognizer(use_ai=True)
                self.ai_enabled = True
            except Exception as e:
                print(f"⚠️ AI initialization failed: {e}")
                self.ai_enabled = False
        else:
            self.ai_enabled = False
        
        # Optimize OpenCV
        cv2.setUseOptimized(True)
        cv2.setNumThreads(4)
        
        # Performance tracking
        self.frame_count = 0
        self.fps = 0
        self.start_time = time.time()
        self.process_every_n_frames = 3  # Process AI every N frames for performance
        self.current_frame = 0
        
        print(f"✅ Enhanced analyzer ready! (AI: {'ON' if self.ai_enabled else 'OFF'})")

    def detect_faces_fast(self, gray_frame):
        """Fast face detection"""
        faces = self.face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces

    def enhanced_gender_age_estimate(self, face_roi, face_rect):
        """
        AI-POWERED gender and age estimation with fallback to heuristics
        """
        x, y, w, h = face_rect
        gray_face = face_roi
        
        # Extract multiple facial features for better estimation
        features = self._extract_facial_features(gray_face, w, h)
        
        # IMPROVED GENDER ESTIMATION (heuristic-based)
        gender = self._estimate_gender(features)
        
        # AI-POWERED AGE ESTIMATION
        if self.ai_enabled and self.current_frame % self.process_every_n_frames == 0:
            age_range, exact_age, confidence = self.age_recognizer.predict_age(gray_face, face_rect)
            age_group = f"{age_range} ({exact_age}y)"
            features['ai_age'] = exact_age
            features['ai_confidence'] = confidence
        else:
            # Fallback to heuristic age estimation
            age_group = self._estimate_age(features)
            features['ai_age'] = None
            features['ai_confidence'] = 0.0
        
        return gender, age_group, features

    def _extract_facial_features(self, gray_face, face_width, face_height):
        """Extract multiple facial features for better analysis"""
        h, w = gray_face.shape
        
        features = {
            'aspect_ratio': w / h,
            'face_area': w * h,
            'jaw_width': w,
            'forehead_height': int(h * 0.25),
            'eye_region_height': int(h * 0.3),
            'face_symmetry': 0.0
        }
        
        # Calculate face symmetry (more symmetric = often perceived as younger)
        try:
            left_half = gray_face[:, :w//2]
            right_half = gray_face[:, w//2:]
            right_half_flipped = cv2.flip(right_half, 1)
            
            if left_half.shape == right_half_flipped.shape:
                diff = cv2.absdiff(left_half, right_half_flipped)
                features['face_symmetry'] = 1.0 - (np.mean(diff) / 255.0)
        except:
            pass
        
        # Detect facial contrast (older faces often have more contrast/wrinkles)
        features['contrast'] = np.std(gray_face)
        
        return features

    def _estimate_gender(self, features):
        """Improved gender estimation with multiple factors"""
        score = 0
        
        # Aspect ratio: wider faces often perceived as masculine
        if features['aspect_ratio'] > 0.85:
            score += 2
        elif features['aspect_ratio'] < 0.75:
            score -= 2
        
        # Jaw width: wider jaw = more masculine
        if features['jaw_width'] > features['face_area'] ** 0.5 * 0.9:
            score += 1
        
        # Face area: larger faces often masculine
        if features['face_area'] > 8000:
            score += 1
        elif features['face_area'] < 4000:
            score -= 1
        
        # Determine gender based on score
        if score >= 1:
            return "Man"
        elif score <= -1:
            return "Woman"
        else:
            return "Person"

    def _estimate_age(self, features):
        """MUCH BETTER age estimation using multiple features"""
        age_score = 0
        
        # Face area: larger faces often older
        if features['face_area'] > 12000:
            age_score += 2  # Older
        elif features['face_area'] < 5000:
            age_score -= 1  # Younger
        
        # Contrast: higher contrast may indicate wrinkles/older
        if features['contrast'] > 50:
            age_score += 1
        elif features['contrast'] < 30:
            age_score -= 1
        
        # Symmetry: more symmetric often perceived as younger
        if features['face_symmetry'] > 0.8:
            age_score -= 1  # Younger
        elif features['face_symmetry'] < 0.6:
            age_score += 1  # Older
        
        # Aspect ratio: changes with age
        if features['aspect_ratio'] > 0.9:
            age_score += 1
        
        # Determine age group based on score
        if age_score >= 3:
            return "60+ Senior"
        elif age_score >= 2:
            return "50-59 Middle"
        elif age_score >= 1:
            return "40-49 Adult"
        elif age_score == 0:
            return "30-39 Adult"
        elif age_score >= -1:
            return "25-29 Young"
        elif age_score >= -2:
            return "18-24 Young"
        else:
            return "0-17 Teen"

def initialize_camera_enhanced():
    """Initialize camera for enhanced detection"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return None
    
    # Optimized settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    
    print("✅ Enhanced camera ready!")
    return cap

def draw_enhanced_overlay(frame, faces, results, fps, ai_enabled=False):
    """Draw enhanced information overlay with AI indicators"""
    for i, (x, y, w, h) in enumerate(faces):
        if i < len(results):
            gender, age_group, features = results[i]
            
            # Color coding
            if "Man" in gender:
                color = (0, 255, 0)  # Green
            elif "Woman" in gender:
                color = (255, 0, 255)  # Pink
            else:
                color = (255, 255, 0)  # Yellow
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Enhanced label with AI indicator
            ai_indicator = "🤖" if features.get('ai_age') else "📊"
            label = f"{ai_indicator} {gender}, {age_group}"
            
            # Text background
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame, (x, y - text_size[1] - 10), 
                         (x + text_size[0], y), color, -1)
            
            # Main label
            cv2.putText(frame, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Additional info with confidence
            if features.get('ai_confidence', 0) > 0:
                confidence = features['ai_confidence']
                info_text = f"Confidence: {confidence:.0%}"
                cv2.putText(frame, info_text, (x, y + h + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    # Performance overlay
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"Faces: {len(faces)}", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    mode_text = "AI-ENHANCED" if ai_enabled else "HEURISTIC"
    cv2.putText(frame, mode_text, (10, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

def main_enhanced():
    """Main AI-enhanced ultra-fast loop"""
    analyzer = EnhancedFaceAnalyzer(use_ai=True)
    cap = initialize_camera_enhanced()
    
    if cap is None:
        return
    
    print("\n" + "="*60)
    print("🤖 AI-ENHANCED ULTRA-FAST FACE ANALYSIS")
    print("="*60)
    print("Now with AI-powered realistic age detection!")
    print("Controls:")
    print("  • Q = Quit")
    print("  • S = Save screenshot") 
    print("  • 1/2/3 = Change resolution")
    print("  • D = Toggle debug info")
    print("  • A = Toggle AI mode")
    print("="*60)
    
    show_debug = False
    
    try:
        while True:
            start_time = time.time()
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale for fast processing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            
            # Fast face detection
            faces = analyzer.detect_faces_fast(gray)
            
            # Enhanced analysis
            results = []
            for face_rect in faces:
                x, y, w, h = face_rect
                face_roi = gray[y:y+h, x:x+w]
                
                if face_roi.size > 0:
                    result = analyzer.enhanced_gender_age_estimate(face_roi, face_rect)
                    results.append(result)
            
            # Increment frame counter
            analyzer.current_frame += 1
            
            # Draw enhanced overlay
            draw_enhanced_overlay(frame, faces, results, analyzer.fps, analyzer.ai_enabled)
            
            # Show debug information
            if show_debug and results:
                debug_y = 120
                for i, (gender, age, features) in enumerate(results):
                    debug_text = f"Face {i+1}: AR{features['aspect_ratio']:.2f} C{features['contrast']:.1f} S{features['face_symmetry']:.2f}"
                    cv2.putText(frame, debug_text, (10, debug_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    debug_y += 15
            
            # Display
            cv2.imshow('🎯 Enhanced Ultra-Fast Face Analysis', frame)
            
            # Update FPS
            analyzer.frame_count += 1
            if analyzer.frame_count >= 15:
                end_time = time.time()
                analyzer.fps = analyzer.frame_count / (end_time - analyzer.start_time)
                analyzer.frame_count = 0
                analyzer.start_time = end_time
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"ai_enhanced_capture_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"💾 Saved: {filename}")
            elif key == ord('d'):
                show_debug = not show_debug
                print(f"🐛 Debug info: {'ON' if show_debug else 'OFF'}")
            elif key == ord('a'):
                analyzer.ai_enabled = not analyzer.ai_enabled
                print(f"🤖 AI mode: {'ON' if analyzer.ai_enabled else 'OFF'}")
                if analyzer.ai_enabled and hasattr(analyzer, 'age_recognizer'):
                    analyzer.age_recognizer.clear_cache()
            elif ord('1') <= key <= ord('3'):
                resolutions = [(320, 240), (640, 480), (800, 600)]
                res_index = key - ord('1')
                new_w, new_h = resolutions[res_index]
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, new_w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, new_h)
                print(f"📐 Resolution: {new_w}x{new_h}")
    
    except KeyboardInterrupt:
        print("\n⏹️ Stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ AI-Enhanced application closed!")

if __name__ == "__main__":
    main_enhanced()