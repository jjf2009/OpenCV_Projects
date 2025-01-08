# required libraries
import cv2 as cv
import mediapipe as mp 
import matplotlib.pyplot as plt
import time 


cap=cv.VideoCapture(0)
# Initializing the Pose and Drawing modules of MediaPipe.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

pTime=0
cTime=0


while True :
    req,img =cap.read()
    # Getting the image's width and height.
    img_width = img.shape[1]
    img_height = img.shape[0]
    imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    result = pose.process(imgRGB)
    if result.pose_landmarks:
           #Specifies the drawing radius of the circles 
           circle_radius = int(.007 * img_height)
           # Specifies the drawing style for landmark connections.
           line_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            # Specifies the drawing style for the 'landmarks'.
           point_spec = mp_drawing.DrawingSpec(color=(220, 100, 0), thickness=-1, circle_radius=circle_radius)
           # Draws both the landmark points and connections.
           mp_drawing.draw_landmarks(
            img,
            landmark_list=result.pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=point_spec,
            connection_drawing_spec=line_spec
            )


    cTime=time.time()
    fps =1/(cTime-pTime)
    pTime=cTime
    cv.putText(img,str(int(fps)),(10,70),cv.FONT_HERSHEY_TRIPLEX,2,(255,0,255),3)
    cv.imshow("Frame",img)
   # Exit the loop on pressing 'q'
    if cv.waitKey(25) & 0xFF == ord('q'):
        break

# Release the VideoCapture object
cap.release()
cv.destroyAllWindows()