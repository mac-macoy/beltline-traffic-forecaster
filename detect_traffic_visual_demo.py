# Code adapted from this script:
# https://gist.github.com/madhawav/1546a4b99c8313f06c0b2d7d7b4a09e2
# which was referenced in this tutorial:
# https://medium.com/@madhawavidanapathirana/real-time-human-detection-in-computer-vision-part-2-c7eda27115c6

# This script is a visual demo of the human detection performed by the model

import cv2
from detector_api import DetectorAPI


model_path = 'models/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
human_detector_api = DetectorAPI(path_to_ckpt=model_path)
threshold = 0.7
cap = cv2.VideoCapture('data/sample.mov')

while True:
    r, img = cap.read()
    img = cv2.resize(img, (1280, 720))

    boxes, scores, classes, num = human_detector_api.process_frame(img)

    human_count = len([score for score in scores if score > threshold])
    print('Human count:', human_count)

    # Visualization of the results of a detection.

    for i in range(len(boxes)):
        # Class 1 represents human
        if classes[i] == 1 and scores[i] > threshold:
            box = boxes[i]
            cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)

    cv2.imshow("preview", img)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
