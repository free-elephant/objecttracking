from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2
import pyrealsense2 as rs
import numpy as np
from socket import *
import threading
import time

def cam_start(sock, vs):
    initBB = None
    while True:
        ret, frames = vs.read()
        frame = np.asarray(frames[0])
        if frame is None:
            break
        frame = imutils.resize(frame, width=500)
        (H, W) = frame.shape[:2]
        if initBB is not None:
            # grab the new bounding box coordinates of the object
            (success, box) = tracker.update(frame)
            # check to see if the tracking was a success
            if success:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (int(x + w / 2), int(y + h / 2)), 1, (0, 255, 0), 2)
                point = (int(x + w / 2), int(y + h / 2))
                point_dis = (vs.depth_frame.get_distance(int(x + w / 2), int(y + h / 2)),0)  # 점으로 거리정보 받기
                coordinate = "coordinate(x,y) " + str(point)
                point = "distance " + str(point_dis)
                sock.send(coordinate.encode('utf-8'))
                sock.send(point.encode('utf-8'))
            # update the FPS counter
            #    if 230<(x+w/2)<250 : #객체가 우측으로 치우쳤을때
            #        #대충 우회전하는 코드
            #    elif (x+w/2) < (vs.WIDTH)/2 - 20 : #객체가 좌측으로 치우쳤을때
            #        #대충 좌회전하는 코드
            #    elif ((x+w/2) < (vs.WIDTH)/2 + 20) and ((x+w/2) > (vs.WIDTH)/2 - 20): #중앙에서 20px 사이 안에 객체의 중앙점이 위치할때
            #        #대충 직진하는 코드
            #    elif point
            fps.update()
            fps.stop()
            # initialize the set of information we'll be displaying on
            # the frame
            info = [
                ("Tracker", args["tracker"]),
                ("Success", "Yes" if success else "No"),
                ("FPS", "{:.2f}".format(fps.fps())),
                ("point", point_dis)
            ]
            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            # select the bounding box of the object we want to track (make
            # sure you press ENTER or SPACE after selecting the ROI)
            initBB = cv2.selectROI("Frame", frame, fromCenter=False,
                                   showCrosshair=True)
            # start OpenCV object tracker using the supplied bounding box
            # coordinates, then start the FPS throughput estimator as well
            tracker.init(frame, initBB)
            fps = FPS().start()
        # if the `q` key was pressed, break from the loop
        elif key == ord("q"):
            break
    # if we are using a webcam, release the pointer
    if not args.get("video", False):
        vs.stop()
    # otherwise, release the file pointer
    else:
        vs.release()
    cv2.destroyAllWindows()
# def send(sock, point, point_dis):
#     while True:
#         coordinate = "coordinate(x,y) " + str(point)
#         point = "distance " + str(point_dis)
#         sock.send(coordinate.encode('utf-8'))
#         sock.send(point.encode('utf-8'))
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--tracker", type=str, default="kcf",
   help="OpenCV object tracker type")
args = vars(ap.parse_args())
OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}
tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
# initialize the bounding box coordinates of the object we are going
# to track
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(0)
time.sleep(1.0)
fps = None
# cam_start(True)
port = 8081
point=0
point_dis=0
clientSock = socket(AF_INET, SOCK_STREAM)
clientSock.connect(('192.168.0.56', port)) #192.168.0.56 141.223.120.154
print('접속 완료')
# sender = threading.Thread(target=send, args=(clientSock, point, point_dis))
cam = threading.Thread(target=cam_start, args=(clientSock, vs, True))
# sender.start()
cam.start()
while True:
    time.sleep(1)
    pass
# if we are using a webcam, release the pointer