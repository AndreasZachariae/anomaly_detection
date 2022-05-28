import json
from typing import List
import requests
import os
import io
import base64
import PIL.Image
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


class MaskRCNN():
    def __init__(self, model_path, label_path, type_name, only_cpu):
        if only_cpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            print("Disabled GPU devices")

        self.model = tf.saved_model.load(model_path)
        self.labels = self.load_json(label_path, type_name)
        print("Trained model loaded")
        print("Classes:", self.labels)

    def load_json(self, label_path, type_name):
        with open(label_path, 'r') as f:
            json_dict = json.load(f)

        return json_dict[type_name]

    def predict(self, image_path, threshold):
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
        # print(infer)
        predictions = infer(tensor)
        # print(predictions["detection_boxes"][0][0])              # tf.Tensor([0.1349324  0.15187985 0.68132347 0.885467  ], shape=(4,), dtype=float32)
        # print(predictions["detection_classes"][0][0])            # tf.Tensor(1.0, shape=(), dtype=float32)
        # print(predictions["detection_masks"][0][0])              # tf.Tensor([[2.5281906e-03 ... 3.3918718e-06] ...], shape=(33, 33), dtype=float32)
        # print(predictions["detection_multiclass_scores"][0][0])  # tf.Tensor([8.9492943e-08 9.9990547e-01 5.3700660e-06 8.9032372e-05], shape=(4,), dtype=float32)
        # print(predictions["detection_scores"][0][0])             # tf.Tensor(0.99990547, shape=(), dtype=float32)
        # print(predictions["image_shape"])                        # tf.Tensor([1.000e+00 1.024e+03 1.024e+03 3.000e+00], shape=(4,), dtype=float32)
        # print(predictions["num_detections"])                     # tf.Tensor([100.], shape=(1,), dtype=float32)
        # print(predictions["num_proposals"])                      # tf.Tensor([300.], shape=(1,), dtype=float32)
        # predictions["mask_predictions"]

        scores = predictions["detection_scores"][0].numpy()
        idx_list = [idx for idx, score in enumerate(scores) if score > threshold]

        detections = []

        for idx in idx_list:
            detections.append({
                "class_id": int(predictions["detection_classes"][0].numpy().astype(np.uint8)[idx]),
                "class_name": self.labels[predictions["detection_classes"][0].numpy().astype(np.uint8)[idx]],
                "probability": float(predictions["detection_scores"][0].numpy()[idx]),
                "bounding_box": predictions["detection_boxes"][0].numpy()[idx],
                "mask": predictions["detection_masks"][0].numpy()[idx]
            })

            # print("Found object: class=" + detections[-1]["class_name"] + ", prob=" + str(detections[-1]["probability"]))

        return detections

    def visualize(self, image_path, detections, annotations=None):
        image = cv2.imread(image_path)

        image = self.draw_bounding_boxes(image, detections)
        # image, mask = self.draw_mask(image, detections)

        if annotations is not None:
            image = self.draw_bounding_boxes(image, annotations, is_true_label=True)
            # image, mask = self.draw_mask(image, annotations)

        # self.show_image(image)
        # self.show_image(mask)
        self.save_image(image, image_path.replace("eval_data", "images_box_overlay"))

        return image

    def draw_bounding_boxes(self, image, detections, is_true_label=False):
        for detection in detections:
            if is_true_label:
                color = (0, 255, 0)
                class_name = detection["class_text"]
                probability = ""
                y1, x1, y2, x2 = self.get_box_coords(image.shape, detection)
                cv2.putText(img=image,
                            text="IoU " + str(round(detection["iou"], 2)),
                            org=(x1+2, y1+25),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1,
                            color=color,
                            thickness=2,
                            lineType=2)

            else:
                color = (0, 0, 255)
                class_name = detection["class_name"]
                probability = " " + str(round(detection["probability"]*100)) + "%"
                x1, y1, x2, y2 = self.get_box_coords(image.shape, detection)

            image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            text = class_name + probability

            cv2.putText(img=image,
                        text=text,
                        org=(x1+2, y1-4),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=color,
                        thickness=2,
                        lineType=2)

        return image

    def draw_mask(self, image, detections):
        image_mask = np.zeros(image.shape, dtype=np.uint8)

        for detection in detections:
            mask = detection["mask"].astype(np.float32)
            # self.show_image(mask)

            x1, y1, x2, y2 = self.get_box_coords(image.shape, detection)

            mask = cv2.resize(mask, (x2-x1, y2-y1))

            # --- detection_masks
            mask_binary = np.zeros(mask.shape)
            mask_binary[mask > .5] = 1

            # --- mask_predictions
            # mask_binary = mask > -10
            # mask_binary = mask_binary.astype(np.uint8)

            mask_binary *= 255
            # self.show_image(mask_binary)

            image_box = image[y1:y2, x1:x2, :]

            alpha = 0.3
            color = (0, 0, 1)
            for c in range(3):
                image_box[:, :, c] = np.where(mask_binary == 255,
                                              image_box[:, :, c] *
                                              (1 - alpha) + alpha * color[c] * 255,
                                              image_box[:, :, c])

            image[y1:y2, x1:x2, :] = image_box
            image_mask[y1:y2, x1:x2, 0] = mask_binary
            image_mask[y1:y2, x1:x2, 1] = mask_binary
            image_mask[y1:y2, x1:x2, 2] = mask_binary

        return image, image_mask

    def get_box_coords(self, image_shape, detection):
        y1 = int(detection["bounding_box"][0] * image_shape[0])
        x1 = int(detection["bounding_box"][1] * image_shape[1])
        y2 = int(detection["bounding_box"][2] * image_shape[0])
        x2 = int(detection["bounding_box"][3] * image_shape[1])

        return x1, y1, x2, y2

    def evaluate(self, images_path, threshold=0.5):
        y_true = []
        y_pred = []
        iou = []
        annotations_dict = self.load_annotation_json(images_path)

        # for class_name in os.listdir(images_path):
        for image_name in os.listdir(os.path.join(images_path)):
            if not image_name.endswith(".jpg"):
                continue
            image_path = os.path.join(images_path, image_name)
            predictions = self.predict(image_path, threshold)

            # -- Use single XML files or one JSON
            # annotations = self.load_annotation_xml(image_path)
            annotations = annotations_dict[image_name.split(".")[0]]

            y_pred_objects, y_true_objects, iou_objects = self.get_true_pred_pair(image_path, annotations, predictions)
            y_true.extend(y_true_objects)
            y_pred.extend(y_pred_objects)
            iou.extend(iou_objects)

            self.visualize(image_path, predictions, annotations)

        accuracy = accuracy_score(y_true, y_pred)
        mean_iou = sum(iou) / len(iou)

        print("accuracy=" + str(accuracy) + " mean_iou=" + str(mean_iou))

        return y_true, y_pred

    def get_true_pred_pair(self, image_path, annotations, predictions):
        y_true = []
        y_pred = []
        iou = []
        good_class_id = self.labels.index("good")
        num_pred = len(predictions)
        num_true = len(annotations)

        if num_true == 0 and num_pred == 0:
            y_true.append(good_class_id)
            y_pred.append(good_class_id)

        if num_pred < num_true:
            max_objects = range(num_true)
        else:
            max_objects = range(num_pred)

        for index in max_objects:
            if index >= num_pred:
                y_pred.append(good_class_id)
                y_true.append(annotations[index]["class_id"])
                iou.append(0)
                annotations[index]["iou"] = 0
            elif index >= num_true:
                y_true.append(good_class_id)
                y_pred.append(predictions[index]["class_id"])
                iou.append(0)
            else:
                y_true.append(annotations[index]["class_id"])
                y_pred.append(predictions[index]["class_id"])
                iou_value = self.intersection_over_union(image_path, annotations[index], predictions[index])
                iou.append(iou_value)
                annotations[index]["iou"] = iou_value

        print(num_pred, num_true, iou)

        return y_pred, y_true, iou

    def intersection_over_union(self, image_path, annotations, predictions):
        image = cv2.imread(image_path)
        true_y1, true_x1, true_y2, true_x2 = self.get_box_coords(image.shape, annotations)
        pred_x1, pred_y1, pred_x2, pred_y2 = self.get_box_coords(image.shape, predictions)

        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(true_x1, pred_x1)
        yA = max(true_y1, pred_y1)
        xB = min(true_x2, pred_x2)
        yB = min(true_y2, pred_y2)

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth rectangles
        boxAArea = (true_x2 - true_x1 + 1) * (true_y2 - true_y1 + 1)
        boxBArea = (pred_x2 - pred_x1 + 1) * (pred_y2 - pred_y1 + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou

    def plot_confusion_matrix(self, y_true, y_pred, metric_path):
        cf_matrix = confusion_matrix(y_true, y_pred, normalize="all")
        accuracy = accuracy_score(y_true, y_pred)
        print(accuracy)
        print(cf_matrix)

        disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix,
                                      display_labels=self.labels)
        # disp.ax_.set_title("Confusion Matrix")

        disp.plot(cmap=plt.cm.Blues)
        path = os.path.join(metric_path, "cf_matrix_acc" + str(round(accuracy*100, 0)) + ".png")
        plt.savefig(path)

    def load_annotation_xml(self, test_image_path):
        xml_path = test_image_path.replace("images", "ground_truth").replace(".png", "_mask.xml")
        tree = ET.parse(xml_path)
        root = tree.getroot()

        annotations = []

        for member in root.findall('object'):
            true_class_name = member[0].text  # class name

            # bbox coordinates
            xmin = int(member[4][0].text)
            ymin = int(member[4][1].text)
            xmax = int(member[4][2].text)
            ymax = int(member[4][3].text)
            # store data in list
            annotations.append({"true_class_id": self.labels.index(true_class_name),
                                "true_class_name": true_class_name,
                                "true_bounding_box": [xmin, ymin, xmax, ymax]})

        return annotations

    def load_annotation_json(self, images_path):
        with open(os.path.join(images_path, "labels.json")) as f:
            data_dict = json.load(f)
            # print(data_dict["0"])
        return data_dict

    def show_image(self, image):
        plt.figure()
        plt.imshow(image)
        plt.show()
        # cv2.imshow("overlays", image)
        # cv2.waitKey(0)

    def save_image(self, image, image_path):
        cv2.imwrite(image_path, image)

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

        cv_mask = self.convert_from_image_to_cv2(masks[0])

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
    type_name = "bottle"
    # images_path = os.path.join(os.getcwd(), "dataset", "augmented_dataset", type_name, "validate", "images")
    images_path = os.path.join(os.getcwd(), "models", "mask_rcnn", "eval_data")
    test_image_path = os.path.join(images_path, "contamination", "013.png")

    model_path = os.path.join(os.getcwd(), "models", "mask_rcnn")

    model = MaskRCNN(model_path=os.path.join(model_path, "saved_model"),
                     label_path=os.path.join(model_path, "labels.json"),
                     type_name=type_name,
                     only_cpu=False)

    # detections = model.predict(test_image_path, threshold=0.5)

    # np.save("./predictions.npy", detections)
    # detections_file = np.load("./predictions.npy", allow_pickle=True)

    # model.visualize(test_image_path, detections_file)

    y_true, y_pred = model.evaluate(images_path)

    model.plot_confusion_matrix(y_true, y_pred, metric_path=model_path)

    # masks = model.infer_on_webserver("http://iras-w06o:9930", os.path.join(path, "broken_small", "001.png"))


if __name__ == '__main__':
    main()
