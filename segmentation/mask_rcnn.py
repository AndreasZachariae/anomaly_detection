from json.tool import main
from tkinter import Image
from typing import List
import requests
import os
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
import io
import base64
import PIL.Image


def main():
    path = os.path.join(os.getcwd(), "dataset", "augmented_dataset", "bottle", "images", "broken_small", "001.png")

    # client_id = 'zaan0001'
    # client_secret = 'anomaly_detection'

    # client = BackendApplicationClient(client_id=client_id)

    image = PIL.Image.open(path)

    output = io.BytesIO()
    image.save(output, format='PNG')
    encoded_bytes = output.getvalue()
    encoded_image = base64.b64encode(encoded_bytes).decode("utf-8")

    data = {'image': encoded_image,
            'model_uid': "anomaly_detection",
            "score_threshold": 0.1}

    result = requests.post("http://iras-w06o:9930/test_infer_model", json=data)

    masks: List[PIL.Image.Image] = []

    for encoded_mask in result.json():
        mask_bytes = io.BytesIO(base64.b64decode(encoded_mask))

        masks.append(PIL.Image.open(mask_bytes))
        masks[-1].show()

    combined = PIL.Image.composite(image, image, masks[0])
    combined.show()

    return masks


if __name__ == '__main__':
    main()
