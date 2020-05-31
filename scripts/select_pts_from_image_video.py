from imutils.video import VideoStream, FileVideoStream
import cv2
import imutils


# vs = VideoStream("").start()
# vs = FileVideoStream("/mnt/sdb1/OwnCloud/datasets/License_Plate_Recognition/Videos/").start()
vs = FileVideoStream("/mnt/sdb1/OwnCloud/datasets/License_Plate_Recognition/Videos/binhduong1.mp4").start()
while True:
    frame = vs.read()
    if frame is None:
        continue

    cv2.imshow("frame", frame)
    key = cv2.waitKey(1) & 0xff
    if key == ord("q"):
        break
    elif key == ord("s"):
        box = cv2.selectROI("frame", frame, fromCenter=False, showCrosshair=True)
        print("x,y,w,h", box)

cv2.destroyAllWindows()
vs.stop()
