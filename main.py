import cv2
import mediapipe as mp
import time
import math
import numpy as np
import sys
import os
import NEW_digit_recog
import csv

def main():
    frameWidth = 1280
    frameHeight = 720
    cap = cv2.VideoCapture(0)
    cap.set(3, frameWidth)
    cap.set(4, frameHeight)
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils

    pTime = 0
    cTime = 0

    sm = 3
    plocX, plocY = 0, 0
    clocX, clocY = 0, 0

    constHd = 35
    wScr, hScr = 1920, 1080

    #set color
    black = (0, 0, 0)
    penSize = 15

    with open('./mathQuest.csv') as f:
        reader = csv.reader(f)
        tempQuestion = [row for row in reader]
    questions = tempQuestion

    def getPoint(hand, index):
        x = hand.landmark[index].x
        y = hand.landmark[index].y
        x = int(x * 1280)
        y = int(y * 720)
        cv2.circle(recognize, (x, y), radius=10, color=black, thickness=-1)
        h, w, c = img.shape
        wCam = 16 * constHd
        hCam = 9 * constHd
        lx = int(np.interp(x - (1280 - wCam) // 2, (0, wCam), (0, wScr)))
        ly = int(np.interp(y - (720 - hCam) // 2, (0, hCam), (0, hScr)))
        return lx, ly


    def paintPoint(lx, ly, color, radius):
        cv2.circle(img, (lx, ly), radius=radius, color=color, thickness=1)

    def distance_cal(x1, y1, x2, y2):
        return int(math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)))

    def calcAns(quest):
        questNow = ''
        ansQuestNow = int(quest[0])
        for i in range(0, len(quest)):
            temp = quest[i]
            questNow += temp
            if temp == ' +':
                ansQuestNow += int(quest[i + 1])
            elif temp == ' x':
                ansQuestNow *= int(quest[i + 1])
            elif temp == ' -':
                ansQuestNow -= int(quest[i + 1])
            elif temp == ' /':
                ansQuestNow /= int(quest[i + 1])
        return  questNow, ansQuestNow

    def solve(a):
        def myFunc(e):
            return e[1];
        a.sort(key=myFunc)
        resString = '';
        for i in a:
            resString += i[0];
        return resString;

    showWindow = True
    checkout = True
    finger = []
    mainColor = (0, 0, 0)
    questNumber = 0
    questNumberNow = 0
    ansQuestNow = 0
    questNow = ''
    popUpFalse = False
    popUpTrue = False
    timePre = time.time()
    timeNow = time.time()
    nextQues = False
    ans = cv2.imread("./ans.png")
    with mpHands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.9) as hands:
        while cap.isOpened():
            # print('true')
            img = cv2.imread('./background.png')
            # print(img)
            if showWindow == False:
                cv2.destroyAllWindow()
                showWindow = True
                time.sleep(2)
            success, recognize = cap.read()
            if not success:
                continue
            recognize = cv2.flip(recognize, 1)
            imgRGB = cv2.cvtColor(recognize, cv2.COLOR_BGR2RGB)

            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if popUpTrue == True:
                img[360:546, 1249:1794] = cv2.imread(f'./TrueAnswer.png')
            if popUpFalse == True:
                img[360:546, 1249:1794] = cv2.imread(f'./WrongAnswer.png')

            timeNow = time.time()

            if timeNow - timePre > 2 and nextQues == True:
                popUpFalse = False
                popUpTrue = False
                questNumberNow += 1
                timePre = timeNow
                nextQues = False
                finger = []

            results = hands.process(imgRGB)
            if questNumber == questNumberNow:
                questNumber += 1
                questNow, ansQuestNow = calcAns(questions[questNumberNow])
            image = cv2.putText(img, questNow, [144, 161], cv2.FONT_HERSHEY_SIMPLEX, 3, black, 4, cv2.LINE_AA)

            for idx in range(0, len(finger) - 1):
                i = finger[idx]
                y = finger[idx + 1]
                if i[4] == True and y[4] == True:
                    cv2.line(img, (i[0], i[1]), (y[0], y[1]), i[2], i[3])

            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    x8, y8 = getPoint(handLms, 8)
                    x4, y4 = getPoint(handLms, 4)
                    xPoint = (x8 + x4) // 2
                    yPoint = (y8 + y4) // 2
                    xPoint = clocX = plocX + (xPoint - plocX) // sm
                    yPoint = clocY = plocY + (yPoint - plocY) // sm
                    plocX, plocY = clocX, clocY
                    paintPoint(xPoint, yPoint, black, round(penSize / 2))

                    range8_4 = distance_cal(x8, y8, x4, y4)

                    x5, y5 = getPoint(handLms, 5)
                    x0, y0 = getPoint(handLms, 0)
                    range0_5 = distance_cal(x0, y0, x5, y5)

                    ratio = range0_5 * 0.32
                    status = False
                    if ratio > range8_4:
                        status = True

                    if status == True:
                        if 198 <= xPoint <= 484 and 402 <= yPoint <= 503:
                            finger = []
                        if 577 <= xPoint <= 1028 and 402 <= yPoint <= 503:
                            ans[602 - 200:1006 - 200, 98:1617] = img[602:1006, 98:1617];
                            scale_percent = 32  # percent of original size
                            width = int(ans.shape[1] * scale_percent / 100)
                            height = int(ans.shape[0] * scale_percent / 100)
                            dim = (width, height)
                            resized = cv2.resize(ans, dim, interpolation=cv2.INTER_AREA)
                            cv2.imwrite("test.png", resized);
                            ansQuest = NEW_digit_recog.main()
                            ansQuest = int(solve(ansQuest))
                            print(ansQuest)
                            timePre = time.time()
                            nextQues = True
                            questNow += ' = ' + str(int(ansQuestNow))
                            if ansQuest == ansQuestNow:
                                # questNumberNow += 1
                                popUpTrue = True
                                popUpFalse = False
                            else:
                                popUpTrue = False
                                popUpFalse = True
                    if finger and finger[-1][4] == False and status == False:
                        continue
                    if 91 <= xPoint <= 1626 and 595 <= yPoint <= 1023:
                        finger.append((xPoint, yPoint, mainColor, penSize, status))
            # Write frame rate
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, "FPS= " + str(int(fps)), (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
            cv2.imshow('image', img)
            if cv2.waitKey(1) == 27:
                break
def tay():
    while True:
        if main():
            main()
        else:
            return

tay()