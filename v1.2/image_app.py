import cv2
import numpy as np
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
import tkinter as tk
from tkinter import filedialog

def get_image_path():
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename(
        title="Select Image", 
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )

def main():
    model = load_model('emotion_detector.h5')
    detector = MTCNN()
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    image_path = get_image_path()
    if not image_path:
        return

    frame = cv2.imread(image_path)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = detector.detect_faces(rgb_frame)

    for face in faces:
        if face['confidence'] < 0.90:
            continue
        x, y, w, h = face['box']
        x, y = max(0, x), max(0, y)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 191, 255), 2)
        
        roi_gray = gray_frame[y:y+h, x:x+w]
        if roi_gray.size == 0:
            continue

        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype('float') / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)

        prediction = model.predict(roi_gray)
        max_index = int(np.argmax(prediction))
        
        # 1. Twarde wymuszenie 1 miejsca po przecinku
        confidence = np.max(prediction) * 100
        text = f"{emotion_labels[max_index]} ({confidence:.1f}%)"

        # 2. Dynamiczne skalowanie czcionki na podstawie szerokości ramki (w)
        font_scale = max(0.4, w / 150.0)
        thickness = max(1, int(font_scale * 2))

        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    cv2.imshow('Emotion Analysis System', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()