from json.tool import main
from typing import List
import requests
import os
import io
import base64
import PIL.Image
import numpy as np
import cv2
import tensorflow as tf

class MaskRCNN():
    def __init__(self, model_path, only_cpu):
        if only_cpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            print("Disabled GPU devices")
            
        self.model = tf.saved_model.load(model_path)
        print("Trained model loaded")
        

    def predict(self, image_path):
        """Wrapper for Keras Model predict function

        Args:
            image_path (Sring): Path to image to predict

        Returns:
            prediction(np.array): array of probabilities for each class
            class_index(int): argmax of prediction array
            class_name(String): name of class index
            image(cv2.Mat): predicted image to display
        """
        image = cv2.imread(image_path)
        tensor = tf.convert_to_tensor(image, dtype=tf.uint8)
        tensor = tf.expand_dims(tensor, 0)

        # print(list(self.model.signatures.keys()))
        infer = self.model.signatures["serving_default"]
        print(infer)
        prediction = infer(tensor)
        print(prediction["class_predictions_with_background"])
        print(prediction["detection_classes"])
        print(prediction["detection_multiclass_scores"])
        print(prediction["detection_scores"])
        print(prediction["detection_masks"])
        print(prediction["num_detections"])


        return prediction, image

    def infer_on_webserver(self, webserver_uri, image_path):
        image = PIL.Image.open(image_path)

        output = io.BytesIO()
        image.save(output, format='PNG')
        encoded_bytes = output.getvalue()
        encoded_image = base64.b64encode(encoded_bytes).decode("utf-8")

        data = {'image': encoded_image,
                'model_uid': "anomaly_detection",
                "score_threshold": 0.1}

        result = requests.post(webserver_uri + "/test_infer_model", json=data)

        masks: List[PIL.Image.Image] = []

        for encoded_mask in result.json():
            mask_bytes = io.BytesIO(base64.b64decode(encoded_mask))

            masks.append(PIL.Image.open(mask_bytes))
            masks[-1].show()

        cv_mask = convert_from_image_to_cv2(masks[0])

        combined = PIL.Image.composite(image, image, masks[0])
        combined.show()

        return masks
        

def convert_from_cv2_to_image(img: np.ndarray):
    # return PIL.Image.fromarray(img)
    return PIL.Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def convert_from_image_to_cv2(img: PIL.Image):
    # return np.asarray(img)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def main():
    path = os.path.join(os.getcwd(), "dataset", "augmented_dataset", "bottle", "images")

    model = MaskRCNN(model_path="models/1/saved_model", only_cpu=True)

    model.predict(os.path.join(path, "broken_small", "001.png"))

    # masks = model.infer_on_webserver("http://iras-w06o:9930", os.path.join(path, "broken_small", "001.png"))

   
if __name__ == '__main__':
    main()
