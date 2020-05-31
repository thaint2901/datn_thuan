import os
pwd_dir = os.path.dirname(os.path.realpath(__file__))

import pickle
import cv2

from .traffic_light_utils import describe_hist, chi2_distance
from config import LIGHT


class TrafficLightRecognize:
    def __init__(self, db_path=os.path.join(pwd_dir, "light_db/traffic_light_db.cpickle")):
        self.db = pickle.load(open(db_path, "rb"))

    
    def recognize(self, image, light_coord=LIGHT):
        x1, y1, x2, y2 = light_coord
        item = image[y1:y2, x1:x2]
        query_features = describe_hist(item)

        results = {}
        for k, features in self.db.items():
            d = chi2_distance(features, query_features)
            results[k] = d
        results = sorted([(v, k) for (k, v) in results.items()])
        light_type = results[0][1]

        return light_type