import cv2
from imutils.video import FileVideoStream
import imutils
import numpy as np

from cameras import FPS, VideoStream
from vehicle import VehicleDetection
from traffic_light.traffic_light_recognize import TrafficLightRecognize
from tracker import Sort
from config import NEG_COORD, POS_COORD, VIDEO_PATH, LIGHT


obj_labels = {
    1: "motor",
    2: "car"
}

light_color = { # for draw traffic_light color
    0: (0, 255, 0), # green
    1: (0, 255, 255), # yellow
    2: (0, 0, 255) # red
}

vs = FileVideoStream(VIDEO_PATH).start()
fps = FPS(nframes=50).start()

vd = VehicleDetection()
tracker = Sort(max_age=10, min_hits=1)
tlr = TrafficLightRecognize()

up = 0
down = 0

while True:
    frame = vs.read()
    if frame is None:
        if isinstance(vs, VideoStream):
            continue
        break

    vehicle_bbs = vd.detect(frame)
    vehicle_bbs = tracker.update(vehicle_bbs)
    light_status = tlr.recognize(frame)
    startX, startY, endX, endY = LIGHT
    cv2.rectangle(frame, (startX, startY), (endX, endY), light_color[light_status], 2)

    for (startX, startY, endX, endY, trk_index) in vehicle_bbs:
        text_label = None
        trackobject = tracker.trackers[trk_index]
        label_id = trackobject.label_id
        obj_id = trackobject.id
        is_passed = trackobject.is_passed
        anomaly = trackobject.anomaly
        light_anomaly = trackobject.light_anomaly

        if is_passed is not None and is_passed is not False:
            trackobject.is_passed = False
            if is_passed < 0:
                up += 1
            else:
                down += 1
            if light_status == 2:
                trackobject.light_anomaly = False
                light_anomaly = False

        vehicle_img = frame[startY:endY, startX:endX]
        if not light_anomaly:
            text_label = "vuot den do"
            cv2.imwrite("images/{}-{}.jpg".format(text_label, trk_index), vehicle_img)
        if not anomaly:
            text_label = "nguoc chieu"
            cv2.imwrite("images/{}-{}.jpg".format(text_label, trk_index), vehicle_img)

        color = trackobject.color
        centroids = trackobject.centroids
        centroids = np.array(centroids, np.int32)
        centroids = centroids.reshape((-1,1,2))
        cv2.polylines(frame, [centroids], False, color, 3)

        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
        label = obj_labels[label_id] + "- ID: {}".format(obj_id)
        cv2.putText(frame, label, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

        if text_label is not None:
            cv2.putText(frame, text_label, (startX, startY-35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    if NEG_COORD is not None:
        startX, startY, endX, endY = NEG_COORD
        cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 255), 2)
    if POS_COORD is not None:
        startX, startY, endX, endY = POS_COORD
        cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 128, 255), 2)

    fps.update()
    text_fps = "FPS: {:.3f}".format(fps.get_fps_n())
    cv2.putText(frame, text_fps, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    counting = "UP: {}, DOWN: {}".format(up, down)
    cv2.putText(frame, counting, (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # if (fps._numFrames) % fps._nframes == 0:
    #     print("FPS: {:.3f}".format(fps.get_fps_n()))

    cv2.imshow("Frame", imutils.resize(frame, width=1200))
    key = cv2.waitKey(1) & 0xff
    if key == ord("q"):
        cv2.destroyAllWindows()
        break

vs.stop()
fps.stop()
