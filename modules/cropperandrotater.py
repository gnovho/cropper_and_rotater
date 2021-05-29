import logging
import os
import traceback

import numpy as np
from cv2 import cv2
from tqdm import tqdm


def max_w_h(arr):
    max_w = max(arr[:, 2])
    max_h = max(arr[:, 3])
    return max_w, max_h

class FaceCropper:
    def cropAndRotateFolderFace(self, folder_input, folder_output):
    threshold = 0.5

    # model
    LBFmodel = "./resources/model_face/lbfmodel.yaml"

    # initialize module to detect facial landmark
    protoPath = "./resources/caffemodel/deploy.prototxt"
    modelPath = "./resources/caffemodel/res10_300x300_ssd_iter_140000.caffemodel"
    images = os.listdir(folder_input)

    for image in tqdm(images):
        try:
            img_path = "./{}/{}".format(folder_input, image)
            cap = cv2.VideoCapture(img_path)
            has_frame, frame = cap.read()
            if has_frame:
                re_img = frame

                for i in range(4): # rotate image 4 times to get front face image
                    re_img = cv2.rotate(re_img, cv2.ROTATE_90_CLOCKWISE)
                    gray = cv2.cvtColor(re_img, cv2.COLOR_BGR2GRAY)

                    # using another module for detect face in image
                    # grab the frame dimensions and construct a blob from the frame
                    (h, w) = gray.shape[:2]
                    # Note must using COLOR IMAGE
                    blob = cv2.dnn.blobFromImage(cv2.resize(re_img, (300, 300)), 1.0,
                                                    (300, 300), (104.0, 177.0, 123.0))

                    # pass the blob through the network and obtain the detections and
                    # predictions
                    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
                    net.setInput(blob)
                    detections = net.forward()

                    # ensure at least one face was found
                    if len(detections) > 0:
                        # we're making the assumption that each image has only ONE
                        # face, so find the bounding box with the largest probability
                        i = np.argmax(detections[0, 0, :, 2])
                        confidence = detections[0, 0, i, 2]
                        # ensure that the detection with the largest probability also
                        # means our minimum probability test (thus helping filter out
                        # weak detections)
                        if confidence > threshold:
                            # compute the (x, y)-coordinates of the bounding box for
                            # the face and extract the face ROI
                            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                            (startX, startY, endX, endY) = box.astype("int")

                            # ensure the detected bounding box does fall outside the
                            # dimensions of the frame
                            startX = max(0, startX)
                            startY = max(0, startY)
                            endX = min(w, endX)
                            endY = min(h, endY)

                            # drop face
                            face = re_img[startY: endY, startX: endX]

                            cv2.imwrite("./{}/out_{}_test_{}.jpg".format(folder_output, image, i), face)
                            # logging.info("./{}/out_{}_test_{}.jpg".format(folder_output, image, i))
        except Exception:
            logging.error("Error at: {} - {}".format(image, traceback._context_message))