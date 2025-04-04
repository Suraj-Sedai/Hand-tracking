import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode = False, maxHands = 2, detectionCon = 0.7, trackCon = 0.7):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon =detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.lmList = []  # Initialize landmark list


    def findHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img


    def findPosition(self, img, handNo = 0, draw = True,):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c, = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        

        return self.lmList

    #functions to count the number of fingers
    def fingersUp(self):
            if len(self.lmList) == 0:
                return []

            fingers = []
            
            # Thumb (compared with index finger base)
            if self.lmList[4][1] > self.lmList[3][1]:  # Thumb extended if x[4] > x[3] (right hand)
                fingers.append(1)
            else:
                fingers.append(0)

            # Other 4 fingers (compared vertically)
            for id in range(8, 21, 4):  # Index, Middle, Ring, Pinky
                if self.lmList[id][2] < self.lmList[id - 2][2]:  # Finger tip is above PIP joint
                    fingers.append(1)
                else:
                    fingers.append(0)

            return fingers

def main():
    pTime = 0
    cTime = 0

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 1280)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        #display how many fingers are up
        fingers = detector.fingersUp()
        totalFingers = fingers.count(1)
        cv2.putText(img, str(totalFingers), (50, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        # cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("VM", img)
        if cv2.waitKey(1) == ord("q"):
            break

if __name__ == "__main__":
    main()