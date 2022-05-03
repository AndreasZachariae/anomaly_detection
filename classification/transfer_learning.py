import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenetv2
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet50
from tensorflow.keras.models import Model


class TransferLearning():
    """Takes the specified model with pretrained weights from imagenet.
    Trains the last layers with custom dataset.
    """

    def __init__(self, model, image_path, image_size, batch_size, only_cpu):
        self.train_dataset, self.val_dataset = self.load_data(image_path, image_size, batch_size)
        self.accuracy = None
        self.loss = None

        if model == "resnet50":
            self.base_model = ResNet50(input_shape=(900, 900, 3), weights='imagenet', include_top=False)
            self.preprocess_input = preprocess_resnet50
        elif model == "mobilenetv2":
            self.base_model = MobileNetV2(input_shape=(900, 900, 3), weights='imagenet', include_top=False)
            self.preprocess_input = preprocess_mobilenetv2
        else:
            print("Model string wrong. Use one of these models:\nresnet50, mobilenetv2")

        if only_cpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    def load_data(self, image_path, image_size, batch_size):
        """Splits dataset in validation and training data 20/80%

        Args:
            image_path (string): Path with all class folders
            image_size (tupel): Size of image
            batch_size (int): Number of images per training step

        Returns:
            train_dataset: 80% of dataset
            val_dataset: Validation 20% of dataset
        """
        train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            image_path,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=image_size,
            batch_size=batch_size)

        val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            image_path,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=image_size,
            batch_size=batch_size)

        return train_dataset, val_dataset

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

    def train(self, learning_rate):
        """Takes the specified model without top layers.
        Add Pooliung and Sense layers to be trained with new data.
        Starts training process.

        Args:
            learning_rate (float): Learning rate to apply
        """
        inputs = tf.keras.Input(shape=(900, 900, 3))
        x = self.preprocess_input(inputs)
        x = self.base_model(x, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        outputs = Dense(4, activation='softmax')(x)

        # Base model und prediction Layer zusammenfügen
        model = Model(inputs=inputs, outputs=outputs)

        # Nur neue Layer sollen trainierbar sein
        for layer in model.layers[:-3]:
            layer.trainable = False
        for layer in model.layers[-3:]:
            layer.trainable = True

        print(model.summary())

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])

        history = model.fit(self.train_dataset,
                            validation_data=self.val_dataset,
                            epochs=5)

        self.accuracy = history.history['accuracy']
        self.loss = history.history['loss']

    def plot_metrics(self):
        """Plot loss and accuracy over epochs
        """
        plt.figure(figsize=(8, 8))
        plt.plot(self.accuracy, label='Training Accuracy')
        plt.plot(self.loss, label='Training Loss')
        plt.legend(loc='lower left')
        plt.ylabel('Accuracy')
        plt.ylim([min(plt.ylim()), 1])
        plt.title('Training Accuracy and Loss')
        plt.show()


def main():
    path = os.path.join(os.getcwd(), "dataset", "augmented_dataset", "bottle", "images")

    model = TransferLearning(model="resnet50",
                             image_path=path,
                             image_size=(900, 900),
                             batch_size=32,
                             only_cpu=False)

    # Best learning rates:
    # mobilenet_v2 = 0.0005
    # resnet50 =

    model.show_example_images()
    model.train(learning_rate=0.0005)
    model.plot_metrics()

    # model.predict(new_image)


if __name__ == "__main__":
    main()
