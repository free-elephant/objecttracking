from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils

import cv2
import pyrealsense2 as rs
import numpy as np

from socket import *
import threading
import time
import sys

class RealsenseCapture:
    def __init__(self):
        self.WIDTH = 640
        self.HEGIHT = 480
        self.FPS = 30
        # Configure depth and color streams
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.WIDTH, self.HEGIHT, rs.format.bgr8, self.FPS)
        self.config.enable_stream(rs.stream.depth, self.WIDTH, self.HEGIHT, rs.format.z16, self.FPS)
    def start(self):
        # Start streaming
        self.pipeline = rs.pipeline()
        self.pipeline.start(self.config)
        print('pipline start')
    def read(self, is_array=True):
        # Flag capture available
        ret = True
        # get frames
        frames = self.pipeline.wait_for_frames()
        # separate RGB and Depth image
        self.color_frame = frames.get_color_frame()  # RGB
        self.depth_frame = frames.get_depth_frame()  # Depth
        if not self.color_frame or not self.depth_frame:
            ret = False
            return ret, (None, None)
        elif is_array:
            # Convert images to numpy arrays
            color_image = np.array(self.color_frame.get_data())
            depth_image = np.array(self.depth_frame.get_data())
            return ret, (color_image, depth_image)
        else:
            return ret, (self.color_frame, self.depth_frame)
    def release(self): # stop
        self.pipeline.stop()


def cam_start(sock,send):
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
                pointx = int(x + w / 2)
                pointy = int(y + h / 2)
                point_dis = vs.depth_frame.get_distance(int(x + w / 2), int(y + h / 2))  # 점으로 거리정보 받기
                coordinate = "coordinate " + str(pointx) + str(pointy)
                point = "distance " + str(point_dis) + str(0)
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

def send(sock, point, point_dis):
    while True:
        coordinate = "coordinate(x,y) " + str(point)
        point = "distance " + str(point_dis)
        sock.send(coordinate.encode('utf-8'))
        sock.send(point.encode('utf-8'))

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
vs = RealsenseCapture()
vs.start()
time.sleep(1.0)
fps = None
# cam_start(True)
try:
    port = 8081
    point=0
    point_dis=0

    # serverSock = socket(AF_INET, SOCK_STREAM)
    # serverSock.connect(('',port))
    # serverSock.listen(1)

    serverSock = socket(AF_INET, SOCK_STREAM)
    serverSock.bind(('',port))
    serverSock.listen(1)

    print('%d waiting' %port)

    connectionSock, addr = serverSock.accept()

    print(str(addr), 'connected')
    sender = threading.Thread(target=send, args=(serverSock, point, point_dis))
    cam = threading.Thread(target=cam_start, args=(serverSock, True))
    sender.start()
    cam.start()
    while True:
        time.sleep(1)
        pass

except KeyboardInterrupt:
    serverSock.close()
    sys.exit()
# if we are using a webcam, release the pointer


