from predict import *
import tensorflow
import cv2
img=cv2.imread('TrainingImage/ Sajal.2.30.jpg')
print(predict(img))