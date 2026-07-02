import cv2
from PIL.ImageChops import offset
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
from mediapipe.tasks.metadata.metadata_schema_py_generated import ImageSize

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5","Model/labels.txt")

offset = 20
ImgSize = 300

folder = "Data/Y"
counter = 0

labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "W", "X", "Y",]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((ImgSize, ImgSize, 3), np.uint8)*255
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]

        ImgCropShape = imgCrop.shape

        aspectRatio = h/w

        if aspectRatio >1:
            k = ImgSize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop,(wCal,ImgSize))
            ImgResizeShape = imgResize.shape
            wGap = math.ceil((ImgSize-wCal)/2)
            imgWhite[:, wGap:wCal+wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite,draw=False)
            print(prediction, index)

        else:
            k = ImgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (ImgSize, hCal ))
            ImgResizeShape = imgResize.shape
            hGap = math.ceil((ImgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite,draw=False)

        cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                      (x - offset+90, y - offset-50+50), (255, 0, 255), 4)
        cv2.putText(imgOutput, labels[index],(x, y-25), cv2.FONT_HERSHEY_SIMPLEX, 1.7,(255,255, 255),2)
        cv2.rectangle(imgOutput, (x-offset, y-offset),
                      (x+w+offset, y+h+offset), (255, 0, 255), 4)


        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)

