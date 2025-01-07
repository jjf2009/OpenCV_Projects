import cv2
import mediapipe as mp
import time 

cap=cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime=0
cTime=0

while True :
    req,img =cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)
    # print(result.multi_hand_landmarks)
    
    if result.multi_hand_landmarks:
        for handlms in result.multi_hand_landmarks:
           for id , lm in enumerate(handlms.landmark):
            #    print(id,lm)
               h,w,c = img.shape
               cx,cv = int(lm.x*w),int(lm.y*h)
               if id  :
                cv2.circle(img,(cx,cv),7,(0,0,255),cv2.FILLED)


           mpDraw.draw_landmarks(img,handlms,mpHands.HAND_CONNECTIONS)
         
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(255,0,255),3)     


    cv2.imshow("Frame",img)
    
    # Exit the loop on pressing 'q'
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()