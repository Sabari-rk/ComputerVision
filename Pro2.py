import cv2 as cv
from Pro import p
import numpy as np
haar=cv.CascadeClassifier('face.xml')
features=np.load('features.npy',allow_pickle=True)
labels=np.load('labels.npy')

face_recognizer=cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img=cv.imread('/home/sabari/Pictures/photo/vijay/download (6).jpeg')
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('ajith',gray)
face_roi=haar.detectMultiScale(gray,1.1,3)
for (x,y,w,h) in face_roi:
    face=gray[y:y+h,x:x+w]
    label,confidence=face_recognizer.predict(face)
    print(p[label],confidence)

    cv.putText(img,p[label],(20,20),cv.FONT_HERSHEY_COMPLEX,1.1,(0,255,0),thickness=2)
    cv.rectangle(img,(x,y),(x+w,y+h),color=(0,255,0),thickness=2)

cv.imshow(p[label],img)
cv.waitKey(0)