# USAGE
# python opencv_object_tracking.py
# python opencv_object_tracking.py --video dashcam_boston.mp4 --tracker csrt
# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2
import pyrealsense2 as rs
import numpy as np
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
    def release(self):
        # Stop streaming
        self.pipeline.stop()


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--tracker", type=str, default="kcf",
   help="OpenCV object tracker type")
args = vars(ap.parse_args())

# extract the OpenCV version info
(major, minor) = cv2.__version__.split(".")[:2]

   # OpenCV object tracker implementations
OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}
   # grab the appropriate object tracker using our dictionary of
   # OpenCV object tracker objects
tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
initBB = None
print("[INFO] starting video stream...")
vs = RealsenseCapture()
vs.start()
time.sleep(1.0)



fps = None
while True:
   ret, frames = vs.read()
   frame = np.asarray(frames[0]) #if args.get("video", False) else frame

   if frame is None:
      break
   # resize the frame (so we can process it faster) and grab the
   # frame dimensions
   frame = imutils.resize(frame, width=500)
   (H, W) = frame.shape[:2]
   # check to see if we are currently tracking an object
   if initBB is not None:
      # grab the new bounding box coordinates of the object
      (success, box) = tracker.update(frame)
      # check to see if the tracking was a success
      if success:
         (x, y, w, h) = [int(v) for v in box]
         cv2.rectangle(frame, (x, y), (x + w, y + h),
            (0, 255, 0), 2)
      # update the FPS counter
      fps.update()
      fps.stop()
      # initialize the set of information we'll be displaying on
      # the frame
      info = [
         ("Tracker", args["tracker"]),
         ("Success", "Yes" if success else "No"),
         ("FPS", "{:.2f}".format(fps.fps())),
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
      initBB = cv2.selectROI("Frame", frame, fromCenter=False,
         showCrosshair=True)
      tracker.init(frame, initBB)
      fps = FPS().start()
   elif key == ord("q"):
      break
# if we are using a webcam, release the pointer
if not args.get("video", False):
   vs.stop()
# otherwise, release the file pointer
else:
   vs.release()
# close all windows
cv2.destroyAllWindows()