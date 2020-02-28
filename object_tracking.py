
import cv2
import numpy as np
import time
import matplotlib.pypSlot as plt
from gtts import gTTS
import string
import speech_recognition as sr

r = sr.Recognizer()
m = sr.Microphone()
test = 1


def checkspeech(r):
    with sr.Microphone() as source:

        audio = r.listen(source)
        try:
            print("You said: " + r.recognize_google(audio))
            return (r.recognize_google(audio))
        except sr.UnknownValueError:
            print("Could not understand audio")
            return ("WW")
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))
            return ("WW")


while (test == 0):

    r.pause_threshold = 0.7
    r.energy_threshold = 1700
    with m as source:
        # r.adjust_for_ambient_noise(source)
        print("Set minimum energy threshold to {}".format(r.energy_threshold))
    print("MICROPHONE")
    speech = str(checkspeech(r))
    print(speech)
    if (speech == "Follow me" or speech == "follow me"):
        test = 1
        print("Following")
    else:
        test = 0

    # speech3=speech.split(' ',1)[0]

drawing = False
ix, iy = -1, -1
R1 = 0
G1 = 0
B1 = 0
b = 0
g = 0
r = 0
set1 = 0
cx1 = 0
cy1 = 0
time1 = time.time()
time2 = 0
a = 0

cap = cv2.VideoCapture(0)
if (test == 1):
    while (cap.isOpened()):
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        # cv2.imshow("frames",frame)
        if (set1 == 0):
            left = 280
            right = 360
            bottom = 280
            top = 200
            kernel = frame[top:bottom, left:right]
            b, g, r, _ = np.uint8(cv2.mean(kernel))
            lb = b - 30
            ub = b + 30
            lg = g - 30
            ug = g + 30
            lr = r - 30
            ur = r + 30
            lower_blue = np.array([lb, lg, lr])
            upper_blue = np.array([ub, ug, ur])
            mask = cv2.inRange(frame, lower_blue, upper_blue)
            dilation = cv2.dilate(mask, (5, 5), iterations=1)
            erosion = cv2.dilate(dilation, (7, 7), iterations=1)
            ret, cnts, heirarchy = cv2.findContours(erosion.astype(np.uint8), cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE)
            set1 = 1
        # cv2.imshow('kernel',kernel)
        mask1 = np.zeros_like(frame)

        if (a == 1):
            # print(frame.shape[0])
            # print(mask1.shape[0])
            # print(frame.shape[1])
            # print(mask1.shape[1])
            lroi = cx - 150
            rroi = cx + 150
            broi = cy + 150
            troi = cy - 150
            if (lroi < 0):
                lroi = 0
            if (rroi > 640):
                rroi = 640
            if (broi > 480):
                broi = 480
            if (troi < 0):
                troi = 0
            mask1[troi:broi, lroi:rroi] = (255, 255, 255)
            masked = cv2.bitwise_and(frame, mask1)
            lower_blue = np.array([lb, lg, lr])
            upper_blue = np.array([ub, ug, ur])
            mask = cv2.inRange(masked, lower_blue, upper_blue)
            dilation = cv2.dilate(mask, (5, 5), iterations=1)
            erosion = cv2.dilate(dilation, (7, 7), iterations=1)
            ret, cnts, heirarchy = cv2.findContours(erosion.astype(np.uint8), cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE)

            # cv2.imshow("A",masked)
            # frame[0:480,0:lroi]=[0,0,0]
            # frame[0:troi,0:640] = [0,0,0]
            # frame[0:480,rroi:640] = [0,0,0]
            # frame[0:640,broi:480] = [0,0,0]
        ##        lower_blue = np.array([lb,lg,lr])
        ##        upper_blue = np.array([ub,ug,ur])
        ##        mask = cv2.inRange(frame, lower_blue, upper_blue)
        ##        dilation = cv2.dilate(mask,(5,5),iterations = 1)
        ##        erosion = cv2.dilate(dilation,(7,7),iterations = 1)
        ##        ret,cnts,heirarchy=cv2.findContours(erosion.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        if len(cnts) != 0:
            maxcnt = max(cnts, key=cv2.contourArea)
            M = cv2.moments(maxcnt)
            if (M['m00'] == 0):
                continue
            else:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.circle(frame, (cx, cy), 10, (255, 255, 255), -1)

        if (set1 == 1):
            time2 = time.time()
            if (time2 - time1 >= 0.5):
                left = cx - 30
                right = cx + 30
                bottom = cy + 30
                top = cy - 30
                a = 1
                kernel = frame[top:bottom, left:right]
                b, g, r, _ = np.uint8(cv2.mean(kernel))
                lb = b - 40
                ub = b + 40
                lg = g - 40
                ug = g + 40
                lr = r - 40
                ur = r + 40
                time1 = time2
                print(lb)

        # cv2.drawContours(frame, approx, -1, (0,255,0), 3)
        res = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.imshow('tester', frame)
        cv2.imshow('tester1', dilation)

        k = cv2.waitKey(1)
        if (k == ord('q')):
            break

cap.release()
cv2.destroyAllWindows()
