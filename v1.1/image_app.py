import cv2
import numpy as np
from tensorflow.keras.models import load_model
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
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    image_path = get_image_path()
    if not image_path:
        return

    frame = cv2.imread(image_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=10)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 191, 255), 2)
        
        roi_gray = gray[y:y+h, x:x+w]
        
        if roi_gray.size == 0:
            continue

        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype('float') / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)

        prediction = model.predict(roi_gray)
        max_index = int(np.argmax(prediction))
        
        confidence = np.max(prediction) * 100
        text = f"{emotion_labels[max_index]} ({confidence:.1f}%)"

        font_scale = max(0.4, w / 150.0)
        thickness = max(1, int(font_scale * 2))

        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    cv2.imshow('Emotion Analysis System', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()