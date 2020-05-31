from filterpy.kalman import KalmanFilter
import numpy as np

from config import NEG_COORD, POS_COORD, HEIGHT_RULE
from .utils import convert_bbox_to_z, convert_x_to_bbox, ray_tracing_numpy


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [
                             0, 0, 0, 1, 0, 0, 0],  [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [
                             0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        x1, y1, x2, y2, label_id = bbox
        cX = int((x1 + x2) / 2.0)
        cY = int((y1 + y2) / 2.0)
        self.centroids = [(cX, cY)]
        self.color = np.random.randint(0, 255, (3,)).tolist()
        self.label_id = label_id
        self.direction = 0 # dung yen
        self.is_passed = None
        self.anomaly = True
        self.light_anomaly = True


    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

        x1, y1, x2, y2, label_id = bbox
        cX = int((x1 + x2) / 2.0)
        cY = int((y1 + y2) / 2.0)
        self.direction = cY - np.mean([c[1] for c in self.centroids])
        self.centroids.append((cX, cY))

        # set is_passed for couting
        fy = self.centroids[0][1]
        is_passed = (fy - HEIGHT_RULE) * (cY - HEIGHT_RULE) < 0
        if is_passed and self.is_passed is None:
            self.is_passed = self.direction
        # if abs(self.direction) > 100 and self.is_passed is None:
        #     self.is_passed = self.direction

        # check anomaly
        if self.anomaly:
            if NEG_COORD is not None:
                startX, startY, endX, endY = NEG_COORD
                poly = [(startX, startY), (endX, startY), (endX, endY), (startX, endY)]
                inside_polygon = ray_tracing_numpy(np.array([cX]), np.array([cY]), poly=poly)[0]
                if inside_polygon and self.direction > 50:
                    self.anomaly = False

            if POS_COORD is not None:
                startX, startY, endX, endY = POS_COORD
                poly = [(startX, startY), (endX, startY), (endX, endY), (startX, endY)]
                inside_polygon = ray_tracing_numpy(np.array([cX]), np.array([cY]), poly=poly)[0]
                if inside_polygon and self.direction < -50:
                    self.anomaly = False


    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6]+self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)