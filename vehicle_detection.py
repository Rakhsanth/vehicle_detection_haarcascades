# -*- coding: utf-8 -*-

import cv2
print(cv2.__version__)

#Trained haar cascade file..
cascade_src = 'cars.xml'
video_src = 'dataset/video1.avi'
#video_src = 'dataset/video2.avi'
#For capturing from video file
cap = cv2.VideoCapture(video_src)
car_cascade = cv2.CascadeClassifier(cascade_src)

while True:
    ret, frame = cap.read()
    if (type(frame) == type(None)):
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)

    for (x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)      
    
    cv2.imshow('video', frame)
    
    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()
