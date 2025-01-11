import cv2 as cv
import mediapipe as mp
import numpy as np
import hand_tracker_modal as htm
import math
import time 
import os

#################
wCam , hCam = 470,1020
################

cap = cv.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
pTime=0

detector = htm.handDetector(detectionCon=0.75)


folderPath ='image'
myList = os.listdir(folderPath)
# print(myList)
overlayList =[]
for imPath in myList:
      image =cv.imread(f"{folderPath}/{imPath}")
#       print(f"{folderPath}/{imPath}")
      overlayList.append(image)

tipIn=[4,8,12,16,20]
totalFingers=0
while cap.isOpened():
    success , img = cap.read()
    if not success:
            print("Ignoring empty camera frame.")
            continue
    
    image.flags.writeable = False
    img=cv.flip(img,1)
    img=detector.findHands(img)
    lmlist = detector.findPosition(img , draw=False)
    if len(lmlist) != 0:
          fingerns =[]

          #Thumb
          if lmlist[tipIn[0]][1] < lmlist[tipIn[0]-1][1]:
                fingerns.append(1)
          else:
                fingerns.append(0)


          #4 fimgers
          for id in range(1,5) :
           if lmlist[tipIn[id]][2] < lmlist[tipIn[id]-2][2]:
                fingerns.append(1)
           else:
                fingerns.append(0)

        #   print(fingerns)        
          totalFingers = fingerns.count(1)
        #   print(totalFingers)
    
    h,w,c= overlayList[0].shape
    img[0:h,0:w] = overlayList[totalFingers]
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    
    cv.putText(img,f'FPS:{int(fps)}',(450,70),cv.FONT_HERSHEY_COMPLEX,1,(255,0,0),1)
    cv.putText(img,f'Num:{int(totalFingers)}',(450,100),cv.FONT_HERSHEY_COMPLEX,1,(255,0,0),1)
    cv.imshow('Face ', img)

    if cv.waitKey(5) & 0xFF == ord('q'):
         break
cap.release()
cv.destroyAllWindows()