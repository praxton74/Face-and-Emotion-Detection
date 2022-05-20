from keras.models import load_model
import cv2
import tensorflow
import numpy as np
model = load_model('emotion.h5')
def predict(img):
    key=['anger', 'surprise', 'disgust', 'fear', 'neutral', 'happiness', 'sadness', 'contempt']
    img = cv2.resize(img, (100, 100))
    img = img.reshape(1, 100, 100, 3)
    res= model.predict(img)
    return (key[np.argmax(res)])