# Code adapted from this script:
# https://gist.github.com/madhawav/1546a4b99c8313f06c0b2d7d7b4a09e2
# which was referenced in this tutorial:
# https://medium.com/@madhawavidanapathirana/real-time-human-detection-in-computer-vision-part-2-c7eda27115c6

# This script processes the camera feed and records the human detection results

import os
import cv2
import time
import boto3
import pandas
import datetime
from detector_api import DetectorAPI


model_path = 'models/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'

# make detector
human_detector_api = DetectorAPI(path_to_ckpt=model_path)

# confidence threshold to count the detected object as a human
threshold = 0.7

# read the video stream
cap = cv2.VideoCapture(1)

# s3
s3_bucket = os.environ.get('S3_BUCKET')
s3_access_key = os.environ.get('S3_ACCESS_KEY')
s3_secret_key = os.environ.get('S3_SECRET_KEY')
s3 = boto3.client('s3', aws_access_key_id=s3_access_key, aws_secret_access_key=s3_secret_key)


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


def write_data_to_s3(df):
    # write the human count data to S3
    s3_key = datetime.datetime.now().strftime('%Y_%m_%d__%H') + '.csv'
    s3.put_object(
        Body=df.to_csv(sep=",", index=False, header=True),
        Bucket=s3_bucket,
        Key=s3_key
    )
    print(f'Wrote data to s3://{s3_bucket}/{s3_key}')


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
        traffic_data_df = pandas.DataFrame(traffic_data, columns=traffic_data_schema)
        write_data_to_s3(traffic_data_df)
        last_written = time.time()
        traffic_data = []

    # we only want a record about every 30 seconds
    time.sleep(29)
