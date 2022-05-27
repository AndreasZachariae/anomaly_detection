import os
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenetv2
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet50
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as preprocess_inceptionresnetv2
from tensorflow.keras.models import Model, load_model
import json
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np


class TransferLearning():
    """Takes the specified model with pretrained weights from imagenet.
    Trains the last layers with custom dataset.
    """

    def __init__(self, model_path, type_name, base_model=None, load_model_name=None, only_cpu=False):
        """Initialize model with base model or use pretrained model if path is given

        Args:
            base_model (String): Name of base model to use
            only_cpu (Bool): Flag to set CUDA_VISIBLE_DEVICES
            model_path (String, optional): Path to pretrained model. Defaults to None.
        """
        if only_cpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            print("Disabled GPU devices")

        self.model_path = model_path
        self.labels = self.load_json(type_name)

        if load_model_name is None:
            if base_model is None:
                print("base_model name can not be None if load_model_name is also None")
            self.model = self.create_model(base_model, type_name, len(self.labels))
            print("Use pretrained base model and train new layers")
            print(self.model.summary())
        else:
            self.model = load_model(os.path.join(self.model_path, load_model_name))
            print("Trained model loaded: " + load_model_name)

        print(self.labels)

        self.train_dataset = None
        self.val_dataset = None
        self.accuracy = []
        self.loss = []
        self.val_accuracy = []
        self.val_loss = []

    def load_data(self, image_path, image_size, batch_size):
        """Splits dataset in validation and training data default=20/80%

        Args:
            image_path (string): Path with all class folders
            image_size (tupel): Size of image
            batch_size (int): Number of images per training step
            validation_split (float): Percentage to split between test and train data
        """
        self.train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            os.path.join(image_path, "train", "images"),
            seed=123,
            image_size=image_size,
            batch_size=batch_size)

        self.val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            os.path.join(image_path, "test", "images"),
            seed=123,
            image_size=image_size,
            batch_size=batch_size)

        self.labels = self.train_dataset.class_names

    def load_json(self, type_name):
        with open(os.path.join(self.model_path, "labels.json"), 'r') as f:
            json_dict = json.load(f)

        return json_dict[type_name]

    def create_model(self, base_model_name, type_name, num_classes):
        """Creates the model from specified pretrained base model without top layer.
        Add two fully connected layers to train with dataset on custom classes

        Args:
            base_model (String): Model to load from tf.keras.applications

        Returns:
            Keras Model: Complete model to be trained
        """
        if base_model_name == "resnet50":
            base_model = ResNet50(input_shape=(900, 900, 3), weights='imagenet', include_top=False)
            preprocess_input = preprocess_resnet50
        elif base_model_name == "mobilenetv2":
            base_model = MobileNetV2(input_shape=(900, 900, 3), weights='imagenet', include_top=False)
            preprocess_input = preprocess_mobilenetv2
        elif base_model_name == "inceptionresnetv2":
            base_model = InceptionResNetV2(input_shape=(900, 900, 3), weights='imagenet', include_top=False)
            preprocess_input = preprocess_inceptionresnetv2
        else:
            print("Model string wrong. Use one of these models:\nresnet50, mobilenetv2, inceptionresnetv2")

        inputs = tf.keras.Input(shape=(900, 900, 3))
        x = preprocess_input(inputs)
        x = base_model(x, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.2)(x)
        x = Dense(1024, activation='relu')(x)
        outputs = Dense(num_classes, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs, name=base_model_name + "_" + type_name)

        base_model.trainable = False

        return model

    def show_example_images(self):
        """Plots a figure with 3*3 example images from the training dataset
        """
        class_names = self.train_dataset.class_names

        plt.figure(figsize=(10, 10))
        for images, labels in self.train_dataset.take(1):
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(class_names[labels[i]])
                plt.axis("off")

        plt.show()

    def train(self, learning_rate, epochs):
        """Compile the model and train on dataset with specified learning rate.

        Args:
            learning_rate (float): Learning rate to apply
            epochs (int): Number of training epochs
        """

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                           metrics=['accuracy'])

        history = self.model.fit(self.train_dataset,
                                 validation_data=self.val_dataset,
                                 epochs=epochs)

        self.accuracy.extend(history.history['accuracy'])
        self.loss.extend(history.history['loss'])
        self.val_accuracy.extend(history.history['val_accuracy'])
        self.val_loss.extend(history.history['val_loss'])

    def fine_tuning(self, learning_rate, epochs):
        self.model.trainable = True
        self.train(learning_rate, epochs)

    def plot_metrics(self, metric_path=None):
        """Plot loss and accuracy over epochs
        """
        plt.figure(figsize=(8, 8))
        plt.plot(self.val_accuracy, label='Validation Accuracy', marker="x", color="blue")
        plt.plot(self.accuracy, label='Training Accuracy', marker="x", color="lightblue")
        plt.plot(self.val_loss, label='Validation Loss', marker="o", color="red")
        plt.plot(self.loss, label='Training Loss', marker="o", color="lightcoral")
        plt.legend(loc='center right')
        plt.ylabel('Accuracy and Loss')
        plt.xlabel('Epochs')
        plt.ylim([min(plt.ylim()), 1])
        plt.title("Model: " + self.model.name)
        if metric_path is None:
            plt.show()
        else:
            path = os.path.join(metric_path, self.model.name + "_acc" + str(round(self.val_accuracy[-1]*100)))
            plt.savefig(path)

    def plot_confusion_matrix(self, metric_path):
        y_pred = []
        y_true = []

        for x, y in self.val_dataset:
            if len(x) > 1:
                print("Batch size has to be 1")
                return
            predictions = self.model.predict(x, verbose=1)
            y_pred.append(tf.math.argmax(predictions, 1).numpy()[0])
            y_true.append(y.numpy()[0])

        cf_matrix = confusion_matrix(y_true, y_pred, normalize="all")
        # cf_matrix = np.array([[0.15706806, 0.11518325, 0.,         0.],
        #                       [0.,         0.19371728, 0.06282723, 0.01570681],
        #                       [0.,   0.0104712, 0.20418848, 0.05759162],
        #                       [0.,    0.,     0.,    0.18324607]])
        accuracy = accuracy_score(y_true, y_pred)
        print(cf_matrix)
        print(accuracy)

        disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix,
                                      display_labels=self.labels)
        # disp.ax_.set_title("Confusion Matrix")

        disp.plot(cmap=plt.cm.Blues)
        path = os.path.join(metric_path, self.model.name + "_matrix_acc" + str(round(accuracy*100)) + ".png")
        plt.savefig(path)

    def save_model(self):
        """Wrapper for Keras Model save function

        Args:
            model_name (String): Name to save model.h5 file
        """
        path = os.path.join(self.model_path, self.model.name + "_acc" + str(round(self.val_accuracy[-1]*100)) + ".h5")
        self.model.save(path, save_format='h5')

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
        image = cv2.resize(image, (900, 900))
        tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        tensor = tf.expand_dims(tensor, 0)
        prediction = self.model.predict(tensor)
        class_index = int(tf.math.argmax(prediction, 1))
        class_name = self.labels[class_index]

        return prediction, class_index, class_name, image

    def evaluate(self):
        """Test model on evaluation dataset

        Returns:
            dict: dict with two keys: accuracy, loss
        """
        result = self.model.evaluate(self.val_dataset, return_dict=True)
        self.val_accuracy = result['accuracy']
        self.val_loss = result['loss']

        return result


def main():
    load_model = True
    types = ["bottle", "hazelnut"]

    for type_name in types:
        images_path = os.path.join(os.getcwd(), "dataset", "augmented_dataset", type_name)
        model_path = os.path.join(os.getcwd(), "models", "transfer_learning", type_name)

        if load_model:
            models = [model_name for model_name in os.listdir(model_path) if model_name.endswith(".h5")]
            for model_name in models:
                model = TransferLearning(type_name=type_name,
                                         model_path=model_path,
                                         load_model_name=model_name,
                                         only_cpu=True)

                # if type_name == "bottle":
                #     test_image_path = os.path.join(images_path, "validate", "images", "contamination", "013.png")
                # else:
                #     test_image_path = os.path.join(images_path, "validate", "images", "print", "001.png")

                # prediction, class_index, class_name, image = model.predict(test_image_path)
                model.load_data(image_path=images_path,
                                image_size=(900, 900),
                                batch_size=1)

                # model.evaluate()
                model.plot_confusion_matrix(metric_path=model_path)

                # print(prediction)
                # cv2.imshow("predicted_class="+class_name, image)
                # cv2.waitKey(0)

        else:
            for model_name, learning_rate, epochs in [("mobilenetv2", 0.001, 10),
                                                      ("resnet50", 0.001, 10),
                                                      ("inceptionresnetv2", 0.001, 10)]:
                print(model_name, learning_rate, epochs)

                model = TransferLearning(model_path=model_path,
                                         type_name=type_name,
                                         base_model=model_name,
                                         only_cpu=False)

                model.load_data(image_path=images_path,
                                image_size=(900, 900),
                                batch_size=32)

                # model.show_example_images()
                model.train(learning_rate, epochs)
                model.train(learning_rate/5, epochs)
                model.train(learning_rate/25, epochs)

                model.load_data(image_path=images_path,
                                image_size=(900, 900),
                                batch_size=2)

                model.fine_tuning(learning_rate/1000, int(epochs/2))
                model.plot_metrics(metric_path=model_path)
                # model.evaluate()
                model.save_model()

                model.plot_confusion_matrix(metric_path=model_path)


if __name__ == "__main__":
    main()
