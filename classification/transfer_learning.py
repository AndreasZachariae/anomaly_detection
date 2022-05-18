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


class TransferLearning():
    """Takes the specified model with pretrained weights from imagenet.
    Trains the last layers with custom dataset.
    """

    def __init__(self, base_model, only_cpu, model_path=None):
        """Initialize model with base model or use pretrained model if path is given

        Args:
            base_model (String): Name of base model to use
            only_cpu (Bool): Flag to set CUDA_VISIBLE_DEVICES
            model_path (String, optional): Path to pretrained model. Defaults to None.
        """
        if only_cpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            print("Disabled GPU devices")

        if model_path is not None:
            self.model = load_model(model_path)
            print("Trained model loaded")
        else:
            self.model = self.create_model(base_model)
            print("Use pretrained base model and train new layers")
        print(self.model.summary())

        self.train_dataset = None
        self.val_dataset = None
        self.accuracy = []
        self.loss = []
        self.val_accuracy = []
        self.val_loss = []

    def load_data(self, image_path, image_size, batch_size, validation_split=0.2):
        """Splits dataset in validation and training data default=20/80%

        Args:
            image_path (string): Path with all class folders
            image_size (tupel): Size of image
            batch_size (int): Number of images per training step
            validation_split (float): Percentage to split between test and train data
        """
        self.train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            image_path,
            validation_split=validation_split,
            subset="training",
            seed=123,
            image_size=image_size,
            batch_size=batch_size)

        self.val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            image_path,
            validation_split=validation_split,
            subset="validation",
            seed=123,
            image_size=image_size,
            batch_size=batch_size)

    def create_model(self, base_model_name):
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
        outputs = Dense(4, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs, name=base_model_name + "_transfer_learning")

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

        self.accuracy = history.history['accuracy']
        self.loss = history.history['loss']
        self.val_accuracy = history.history['val_accuracy']
        self.val_loss = history.history['val_loss']

    def fine_tuning(self, learning_rate, epochs):
        self.model.trainable = True
        self.train(learning_rate, epochs)

    def plot_metrics(self):
        """Plot loss and accuracy over epochs
        """
        plt.figure(figsize=(8, 8))
        plt.plot(self.accuracy, label='Training Accuracy', marker="x", color="blue")
        plt.plot(self.loss, label='Training Loss', marker="o", color="red")
        plt.plot(self.val_accuracy, label='Validation Accuracy', marker="x", color="lightblue")
        plt.plot(self.val_loss, label='Validation Loss', marker="o", color="lightcoral")
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.ylim([min(plt.ylim()), 1])
        plt.title("Model: " + self.model.name)
        plt.show()

    def save_model(self, model_name):
        """Wrapper for Keras Model save function

        Args:
            model_name (String): Name to save model.h5 file
        """
        path = os.path.join("models", model_name + "_acc" + str(round(self.val_accuracy[-1]*100)) + ".h5")
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
        tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        tensor = tf.expand_dims(tensor, 0)
        prediction = self.model.predict(tensor)
        class_index = int(tf.math.argmax(prediction, 1))

        if self.train_dataset is None:
            class_name = str(class_index)
        else:
            class_name = self.train_dataset.class_names[class_index]

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
    load_model = False
    fine_tuning = False

    path = os.path.join(os.getcwd(), "dataset", "augmented_dataset", "bottle", "images")

    if load_model:
        for file_name, learning_rate, epochs in [("old/inceptionresnetv2_acc95.h5", 0.0001, 10), ("old/mobilenetv2_acc99.h5", 0.00001, 10), ("old/resnet50_acc98.h5", 0.0001, 10)]:
            model = TransferLearning(base_model="",
                                     only_cpu=False,
                                     model_path="models/" + file_name)
            if fine_tuning:
                model.load_data(image_path=path,
                                image_size=(900, 900),
                                batch_size=4,
                                validation_split=0.2)

                model.fine_tuning(learning_rate, epochs)
                model.save_model(model_name)
                model.plot_metrics()

            else:
                prediction, class_index, class_name, image = model.predict(os.path.join(path, "good", "000.png"))
                print(prediction)
                cv2.imshow("predicted_class="+class_name, image)
                cv2.waitKey(0)

    else:
        # Best learning rates:
        # mobilenetv2 = 0.0005
        # resnet50 = 0.001
        # inceptionresnetv2 = 0.005
        for model_name, learning_rate, epochs in [("mobilenetv2", 0.0001, 50), ("inceptionresnetv2", 0.001, 50), ("resnet50", 0.001, 50)]:
            print(model_name, learning_rate, epochs)

            model = TransferLearning(base_model=model_name,
                                     only_cpu=False)

            model.load_data(image_path=path,
                            image_size=(900, 900),
                            batch_size=32,
                            validation_split=0.2)

            # model.show_example_images()
            model.train(learning_rate, epochs)
            model.plot_metrics()
            # model.evaluate()
            model.save_model(model_name)


if __name__ == "__main__":
    main()
