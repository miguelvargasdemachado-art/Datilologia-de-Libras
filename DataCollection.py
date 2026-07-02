import cv2
from PIL.ImageChops import offset
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
from mediapipe.tasks.metadata.metadata_schema_py_generated import ImageSize

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
ImgSize = 300

folder = "Data/Y"
counter = 0

while True:
    success, img = cap.read()
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
        else:
            k = ImgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (ImgSize, hCal ))
            ImgResizeShape = imgResize.shape
            hGap = math.ceil((ImgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize


        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)
