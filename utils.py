import cv2
import numpy as np

def preprocess_face(face_roi):
    """
    Preprocess face ROI for analysis
    """
    if face_roi.size == 0:
        return None
        
    try:
        # Resize face to standard size
        face_resized = cv2.resize(face_roi, (224, 224))
        return face_resized
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return None

def draw_bounding_box(frame, x, y, w, h, label, confidence=None):
    """
    Draw bounding box and label on frame
    """
    try:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Prepare label text
        label_text = label
        if confidence is not None:
            label_text += f' ({confidence:.2f})'
        
        # Calculate text background
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x, y - text_size[1] - 10), 
                     (x + text_size[0], y), (0, 255, 0), -1)
        
        # Put text
        cv2.putText(frame, label_text, (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return frame
    except Exception as e:
        print(f"Drawing error: {e}")
        return frame