import cv2
import os

class FaceExtractor:
    """
    Extracts faces from images using OpenCV's Haar Cascade.
    Can be easily swapped with MTCNN or MediaPipe for higher accuracy.
    """
    def __init__(self, cascade_path=None):
        if cascade_path is None:
            # Use default Haar Cascade included with OpenCV
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        else:
            self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def extract_face(self, image, target_size=(224, 224)):
        """
        Detects, crops, and resizes the primary face from an image.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            # Fallback: if no face detected, use a center crop of the image
            h, w = image.shape[:2]
            size = min(h, w)
            y = (h - size) // 2
            x = (w - size) // 2
            face_crop = image[y:y+size, x:x+size]
            return cv2.resize(face_crop, target_size)

        # Sort by area to get the largest face (assumed primary subject)
        (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
        
        # Crop and resize
        face_crop = image[y:y+h, x:x+w]
        face_resized = cv2.resize(face_crop, target_size)
        
        return face_resized

if __name__ == "__main__":
    # Test logic
    extractor = FaceExtractor()
    # image = cv2.imread('test.jpg')
    # face = extractor.extract_face(image)
    print("FaceExtractor initialized successfully.")
