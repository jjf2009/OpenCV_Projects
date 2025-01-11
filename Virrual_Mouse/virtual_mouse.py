import cv2 as cv
import numpy as np
import hand_tracker_modal as htm
import pyautogui
import time
import math


def main():
    # Set webcam resolution
    wCam, hCam = 640, 480
    cap = cv.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)

    # Initialize hand detector
    detector = htm.handDetector(detectionCon=0.75)

    # Get screen resolution
    wScr, hScr = pyautogui.size()

    # Variables for FPS calculation
    pTime = 0

    frameR =100
    plocX,plocY =0 , 0
    clocX , clocY =0 ,0
    smoothening = 7


    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Error: Failed to capture image.")
            continue

        # Flip and process the frame
        img = cv.flip(img, 1)
        img = detector.findHands(img, draw=True)
        lmlist = detector.findPosition(img, draw=False)
        # 

        if len(lmlist) != 0:
            # Get positions of index and middle finger
            x1, y1 = lmlist[8][1:]  # Index finger tip
            x2, y2 = lmlist[12][1:]  # Middle finger tip
            cx,cy = (x1+x2)//2,(y1+y2)//2
            # Detect which fingers are up
            fingers = detector.FingerUP(img)
            cv.rectangle(img,(frameR,frameR),(wCam-frameR,hCam-frameR),(255,0,255),2)
            # If only the index finger is up, move the mouse
            if len(fingers) >= 2 and fingers[1] == 1 and fingers[2] == 0:
                # Convert coordinates to screen size
                x3 = np.interp(x1, (frameR, wCam-frameR), (frameR, wScr-frameR))
                y3 = np.interp(y1, (frameR, hCam-frameR), (frameR, hScr-frameR))
                
                
                

                # Smooth mouse movement
                clocX = plocX+(x3-plocX)/smoothening
                clocY = plocY + (y3-plocY)/smoothening

                #Move Move 
                pyautogui.moveTo( clocX,  clocY)
                cv.circle(img,(x1,y1),8,(255,0,255),cv.FILLED)
                plocX,plocY = clocX , clocY
            if len(fingers) >= 2 and fingers[1] == 1 and fingers[2] == 1:
                length = math.hypot(x2-x1,y2-y1)
                cv.line(img,(x1,y1),(x2,y2),(255,0,255),2)
                cv.circle(img,(cx,cy),8,(0,0,255),cv.FILLED)  
                if(length<25):
                    pyautogui.click()     
                    cv.circle(img,(cx,cy),8,(0,255,0),cv.FILLED)  

        # Calculate FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime) if cTime - pTime > 0 else 0
        pTime = cTime

        # Display FPS on the frame
        cv.putText(img, f'FPS: {int(fps)}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        # Show the processed frame
        cv.imshow('Virtual Mouse', img)

        # Exit on 'q' key press
        if cv.waitKey(5) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
