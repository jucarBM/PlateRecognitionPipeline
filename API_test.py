import boto3
import requests
import hashlib
import hmac
import cv2
import datetime
import time

your_env_access_key_var = 'AKIAZC5KVF2CCPQ342Z4'
your_env_secret_key_var = 'SvinHqNhKFxcTAhLe6wsNzqHjQ6zngiuWCYamH5h'
your_stream_name = 'ExampleStream'

ARN_kinesis = "arn:aws:kinesisvideo:us-west-2:624738774660:stream/ExampleStream/1624750290639"
ARN_datastream = "arn:aws:kinesis:us-west-2:624738774660:stream/TestVideoStream"
ARN_role = "arn:aws:iam::624738774660:role/Rekognition"

if __name__ == '__main__':

    client = boto3.client('rekognition', 'us-west-2', aws_access_key_id=your_env_access_key_var,
                          aws_secret_access_key=your_env_secret_key_var)

    response = client.create_stream_processor(
        Input={
            'KinesisVideoStream': {
                'Arn': ARN_kinesis
            }
        },
        Output={
            'KinesisDataStream': {
                'Arn': ARN_datastream
            }
        },
        Name='processortest',
        Settings={
            'FaceSearch': {
                'CollectionId': 'string',
                'FaceMatchThreshold': ...
            }
        },
        RoleArn=ARN_role,
        Tags={
            'type': 'test'
        }
    )

