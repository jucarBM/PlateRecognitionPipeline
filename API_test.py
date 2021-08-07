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

ARN = "arn:aws:kinesisvideo:us-west-2:624738774660:stream/ExampleStream/1624750290639"


def get_endpoint_boto():
    client = boto3.client('kinesisvideo', 'us-west-2', aws_access_key_id=your_env_access_key_var,
                          aws_secret_access_key=your_env_secret_key_var)
    response = client.get_data_endpoint(
        StreamName=your_stream_name,
        APIName='PUT_MEDIA'
    )
    print(response)
    endpoint = response.get('DataEndpoint', None)
    print("endpoint %s" % endpoint)
    if endpoint is None:
        raise Exception("endpoint none")
    return endpoint


def sign(key, msg):
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def get_signature_key(key, date_stamp, regionName, serviceName):
    kDate = sign(('AWS4' + key).encode('utf-8'), date_stamp)
    kRegion = sign(kDate, regionName)
    kService = sign(kRegion, serviceName)
    kSigning = sign(kService, 'aws4_request')
    return kSigning


def get_host_from_endpoint(endpoint):
    if not endpoint.startswith('https://'):
        return None
    retv = endpoint[len('https://'):]
    return str(retv)


def get_region_from_endpoint(endpoint):
    if not endpoint.startswith('https://'):
        return None
    retv = endpoint[len('https://'):].split('.')[2]
    return str(retv)


class gen_request_parameters:
    def __init__(self):
        self._data = ''

        if True:
            print('True')
            # cap = cv2.VideoCapture("rtsp://admin:123@172.14.10.1/ch0_0.264")
            self.cap = cv2.VideoCapture(0)
            # cap.set(5, 60)
            if not self.cap.isOpened():
                print('Camera error')
                exit()

    def new_frame(self):
        # ret, frame = self.cap.read()
        request_parameters = self.cap.read()
        print(f'Frame loaded: {request_parameters[0]}')
        return request_parameters


def mediagolive(request_camera, your_stream_name):
    your_stream_name = your_stream_name
    print(your_stream_name)
    endpoint = get_endpoint_boto()

    method = 'POST'
    service = 'kinesisvideo'
    host = get_host_from_endpoint(endpoint)
    region = get_region_from_endpoint(endpoint)
    endpoint += '/putMedia'
    content_type = 'application/json'
    start_tmstp = repr(time.time())
    access_key = None
    secret_key = None
    while True:
        k = your_env_access_key_var
        if k is not None and type(k) is str and k.startswith('AKIA'):
            access_key = k
        k = your_env_secret_key_var
        if k is not None and type(k) is str and len(k) > 4:
            secret_key = k
        break
    if access_key is None or secret_key is None:
        print('No access key is available.')
        exit()
    t = datetime.datetime.utcnow()
    amz_date = t.strftime('%Y%m%dT%H%M%SZ')
    date_stamp = t.strftime('%Y%m%d')
    canonical_uri = '/putMedia'
    canonical_querystring = ''
    canonical_headers = ''
    canonical_headers += 'connection:keep-alive\n'
    canonical_headers += 'content-type:application/json\n'
    canonical_headers += 'host:' + host + '\n'
    canonical_headers += 'transfer-encoding:chunked\n'
    canonical_headers += 'user-agent:AWS-SDK-KVS/2.0.2 GCC/7.4.0 Linux/4.15.0-46-generic x86_64\n'
    canonical_headers += 'x-amz-date:' + amz_date + '\n'
    canonical_headers += 'x-amzn-fragment-acknowledgment-required:1\n'
    canonical_headers += 'x-amzn-fragment-timecode-type:ABSOLUTE\n'
    canonical_headers += 'x-amzn-producer-start-timestamp:' + start_tmstp + '\n'
    canonical_headers += 'x-amzn-stream-name:' + your_stream_name + '\n'
    signed_headers = 'connection;content-type;host;transfer-encoding;user-agent;'
    signed_headers += 'x-amz-date;x-amzn-fragment-acknowledgment-required;'
    signed_headers += 'x-amzn-fragment-timecode-type;x-amzn-producer-start-timestamp;x-amzn-stream-name'
    canonical_request = method + '\n' + canonical_uri + '\n' + canonical_querystring + '\n' + canonical_headers + '\n' + signed_headers
    canonical_request += '\n'
    canonical_request += hashlib.sha256(''.encode('utf-8')).hexdigest()
    algorithm = 'AWS4-HMAC-SHA256'
    credential_scope = date_stamp + '/' + region + '/' + service + '/' + 'aws4_request'
    string_to_sign = algorithm + '\n' + amz_date + '\n' + credential_scope + '\n' + hashlib.sha256(
        canonical_request.encode('utf-8')).hexdigest()

    signing_key = get_signature_key(secret_key, date_stamp, region, service)

    signature = hmac.new(signing_key, (string_to_sign).encode('utf-8'),
                         hashlib.sha256).hexdigest()

    authorization_header = algorithm + ' ' + 'Credential=' + access_key + '/' + credential_scope + ', '
    authorization_header += 'SignedHeaders=' + signed_headers + ', ' + 'Signature=' + signature

    headers = {
        'Accept': '*/*',
        'Authorization': authorization_header,
        'connection': 'keep-alive',
        'content-type': content_type,
        # 'host': host,
        'transfer-encoding': 'chunked',
        # 'x-amz-content-sha256': 'UNSIGNED-PAYLOAD',
        'user-agent': 'AWS-SDK-KVS/2.0.2 GCC/7.4.0 Linux/4.15.0-46-generic x86_64',
        'x-amz-date': amz_date,
        'x-amzn-fragment-acknowledgment-required': '1',
        'x-amzn-fragment-timecode-type': 'ABSOLUTE',
        'x-amzn-producer-start-timestamp': start_tmstp,
        'x-amzn-stream-name': your_stream_name,
        'Expect': '100-continue'
    }

    # ************* SEND THE REQUEST *************
    print('\nBEGIN REQUEST++++++++++++++++++++++++++++++++++++')
    print('Request URL = ' + endpoint)

    r = requests.post(endpoint, data=request_camera.new_frame(), headers=headers)

    print('\nRESPONSE++++++++++++++++++++++++++++++++++++')
    print('Response code: %d\n' % r.status_code)
    print(r.text)


if __name__ == '__main__':
    request_camera = gen_request_parameters()
    mediagolive(request_camera, your_stream_name)
