import cv2 as cv
import mediapipe as mp
import time 
import numpy as np
import hand_tracker_modal as htm
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

#################
wCam , hCam = 470,1020

################

cap = cv.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
pTime = 0

detector = htm.handDetector(detectionCon=0.7)
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange=volume.GetVolumeRange()
# volume.SetMasterVolumeLevel(-20.0, None)
minVol= volRange[0]
maxVol = volRange[1]
vol=0
volBar =488
fps=0
while cap.isOpened():
    success , image = cap.read()
    if not success:
            print("Ignoring empty camera frame.")
            continue
    image =detector.findHands(image)
    lmlist = detector.findPosition(image , draw=False)
    if len(lmlist) != 0:
        #   print(lmlist[4],lmlist[8])

          x1,y1=lmlist[4][1],lmlist[4][2]
          x2,y2 = lmlist[8][1],lmlist[8][2]
          cx,cy = (x1+x2)//2,(y1+y2)//2

          cv.circle(image,(x1,y1),8,(255,0,255),cv.FILLED)     
          cv.circle(image,(x2,y2),8,(255,0,255),cv.FILLED)     
          cv.circle(image,(cx,cy),8,(255,0,255),cv.FILLED)  

          cv.line(image,(x1,y1),(x2,y2),(255,0,255),2)

          length = math.hypot(x2-x1,y2-y1)
        #   print(length)
           # Hand range 50 - 230 
        # Volume Range -63.5 - 0 

          vol = np.interp(length,[50,170],[minVol,maxVol])
          volBar=np.interp(length,[50,150],[488,150])
        #   print(length,vol)
          volume.SetMasterVolumeLevel(vol, None)
     
          if(length<50):
                cv.circle(image,(cx,cy),8,(0,0,255),cv.FILLED)  

       
    
    cv.rectangle(image,(50,150),(85,488),(0,255,0),3)
    cv.rectangle(image,(50,int(volBar)),(85,488),(0,255,0),cv.FILLED)
    image=cv.flip(image,1)
    cv.putText(image,f'FPS:{int(fps)}',(40,50),cv.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
    

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime=cTime
    

    cv.imshow('Face ', image)

    if cv.waitKey(5) & 0xFF == ord('q'):
         break
cap.release()
cv.destroyAllWindows()