# Code adapted from this script:
# https://gist.github.com/madhawav/1546a4b99c8313f06c0b2d7d7b4a09e2
# which was referenced in this tutorial:
# https://medium.com/@madhawavidanapathirana/real-time-human-detection-in-computer-vision-part-2-c7eda27115c6

# This script processes the camera feed and records the human detection results

import cv2
import time
import pandas
import datetime
from detector_api import DetectorAPI


model_path = 'models/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'

# make detector
human_detector_api = DetectorAPI(path_to_ckpt=model_path)

# confidence threshold to count the detected object as a human
threshold = 0.7

# read the video stream
cap = cv2.VideoCapture(0)


def get_human_count():
    # read and resize the video frame for the model
    r, img = cap.read()
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img = cv2.resize(img, (720, 1280))
    img = img[650:-50, 0:-1]

    # process the frame
    boxes, scores, classes, num = human_detector_api.process_frame(img)

    # determine the number of humans from the model result
    human_count = len([score for score in scores if score > threshold])
    return human_count


# array to hold the traffic data
traffic_data = []
traffic_data_schema = ['human_count', 'timestamp']

# variable to notate when to write the data to a csv file
last_written = time.time()

while True:
    # get the human count and timestamp
    human_count = get_human_count()
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    traffic_data.append([human_count, timestamp])

    # write the data to a csv file if an hour has passed
    if time.time() - last_written > 60*60:
        df = pandas.DataFrame(traffic_data, columns=traffic_data_schema)
        csv_file_path = datetime.datetime.now().strftime('%Y_%m_%d__%H') + '.csv'
        df.to_csv(f'traffic_data/{csv_file_path}', index=False)
        last_written = time.time()
        traffic_data = []

    # we only want a record about every 10 seconds
    time.sleep(9)