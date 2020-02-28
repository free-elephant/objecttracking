import cv2
import numpy as np

cap = cv2.VideoCapture(0)

if cap.isOpened() == False:
    print('Camera is closed!')

while cap.isOpened():
    fram = cap.read();

    cv2.imshow('Frame', fram)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

print('END!')
