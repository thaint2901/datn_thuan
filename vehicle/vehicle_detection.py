from tensorflow_serving.apis.predict_pb2 import PredictRequest
import grpc
from tensorflow_serving.apis import prediction_service_pb2_grpc
import tensorflow as tf
import numpy as np
import cv2

from config import SUB_RECT


class VehicleDetection:
    def __init__(self, model_name="VehicleDetector", min_confidence=0.8):
        self.min_confidence = min_confidence

        self.request = PredictRequest()
        self.request.model_spec.name = model_name
        self.request.model_spec.signature_name = "serving_default"
        self.channel = grpc.insecure_channel('localhost:8500')
        self.predict_service = prediction_service_pb2_grpc.PredictionServiceStub(self.channel)

        self.sub_start_x = None
    

    def detect(self, image):
        h, w = image.shape[:2]
        if self.sub_start_x is None:
            self.sub_start_x = int(w * SUB_RECT[0])
            self.sub_start_y = int(h * SUB_RECT[1])
            self.sub_end_x = int(w * SUB_RECT[2])
            self.sub_end_y = int(h * SUB_RECT[3])
        image = image[self.sub_start_y:self.sub_end_y, self.sub_start_x:self.sub_end_x].copy()
        h, w = image.shape[:2]
        

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images = np.array(image[np.newaxis], dtype=np.uint8)

        self.request.inputs['inputs'].CopyFrom(tf.make_tensor_proto(images))
        response = self.predict_service.Predict(self.request, timeout=10.0)
        boxes = tf.make_ndarray(response.outputs["detection_boxes"])
        scores = tf.make_ndarray(response.outputs["detection_scores"])
        labels = tf.make_ndarray(response.outputs["detection_classes"])

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        labels = np.squeeze(labels)

        result = np.zeros((0, 5), dtype=np.int32)
        for (box, score, label) in zip(boxes, scores, labels):
            if score < self.min_confidence:
                continue

            (startY, startX, endY, endX) = box
            startX = int(startX * w) + self.sub_start_x
            startY = int(startY * h) + self.sub_start_y
            endX = int(endX * w) + self.sub_start_x
            endY = int(endY * h) + self.sub_start_y

            result = np.append(result, np.array([[startX, startY, endX, endY, label]], dtype=np.int32), axis=0)

        return result
