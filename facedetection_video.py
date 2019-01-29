import os
import numpy as np
import cv2
import _pickle as pickel

face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt2.xml')

recogniser = cv2.face.LBPHFaceRecognizer_create();
recogniser.read('face_recogniser_model.xml')
labelencoder = pickel.load( open('labelencoder.dat', 'rb'))

cap = cv2.VideoCapture(0)
while True:
    check,frame = cap.read() 
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces_coords = face_cascade.detectMultiScale(frame_gray, scaleFactor = 1.05, minNeighbors = 5)
    for x,y,w,h in faces_coords:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        face = frame_gray[y:y+h, x:x+w]
        face_resized =  cv2.resize(face, (50,50))
        pred , conf = recogniser.predict(face_resized)
        
        if conf<=140:
            prediction = labelencoder.inverse_transform(pred)
        else :
           prediction = 'Unknown'
        cv2.putText(frame, prediction+' ('+str(round(conf,2))+')' ,(int(x+w/2),y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,255), 1)
        
    cv2.imshow('frame',frame)
    key = cv2.waitKey(1)
    
    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()




