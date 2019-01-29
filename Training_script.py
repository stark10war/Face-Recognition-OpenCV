
######### This is the training script ############
# It reads  all the images in the directories and convert them to a numpy array with labels

import os
os.chdir('D:\\Python practice\\OpenCV practice project')
import numpy as np
import pandas as pd
import cv2
import _pickle as pickel



BASE_DIR = os.path.dirname(os.path.abspath('face-train.py'))
image_dir =  os.path.join(BASE_DIR, "Input_images")

y_labels = []
face_train = []

recogniser = cv2.face.LBPHFaceRecognizer_create();

for root, dirs, files in os.walk(image_dir):
    for file in files:
        path = os.path.join(root,file)
        label = os.path.basename(root)
        print(label,path)
        img = cv2.imread(path,0)
        y_labels.append(label)
        face_train.append(img)


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
y_labels = labelencoder.fit_transform(y_labels)
pickel.dump(labelencoder, open('labelencoder.dat', 'wb')) # dumping the model to disk

# Training for all faces
recogniser.train(face_train, y_labels)
recogniser.save('face_recogniser_model.xml')

            
            
