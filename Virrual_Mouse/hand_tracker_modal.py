import cv2
import mediapipe as mp
import time


class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.mpStyles = mp.solutions.drawing_styles

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img,
                        handLms,
                        self.mpHands.HAND_CONNECTIONS,
                        self.mpStyles.get_default_hand_landmarks_style(),
                        self.mpStyles.get_default_hand_connections_style()
                    )
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmlist = []

        if self.results.multi_hand_landmarks:
            if handNo < len(self.results.multi_hand_landmarks):
                myHand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmlist.append([id, cx, cy])
            else:
                print(f"Error: handNo {handNo} is out of range.")
        return lmlist

    def FingerUP(self, img, draw=True):
        lmlist = self.findPosition(img)
        tipIn = [4, 8, 12, 16, 20]
          # Default all fingers to "down" (0)
        if len(lmlist) !=0:  # Ensure 21 landmarks are present
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



        return fingerns


def main():
    pTime = 0
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    detector = handDetector()

    while True:
        ret, img = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        img = detector.findHands(img)
        totalFingers = detector.FingerUP(img)

        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime
        img = cv2.flip(img, 1)
        cv2.putText(img, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0, 255), 3)

        cv2.imshow("Frame", img)

        # Exit the loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture object
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
