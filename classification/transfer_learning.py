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

    def __init__(self, base_model, image_path, image_size, batch_size, only_cpu, model_path=None):
        if only_cpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            print("Disabled GPU devices")

        self.train_dataset, self.val_dataset = self.load_data(image_path, image_size, batch_size)

        if model_path is not None:
            self.model = load_model(model_path)
            print("Trained model loaded")
        else:
            self.model = self.create_model(base_model)
            print("Use pretrained base model and train new layers")
        print(self.model.summary())

        self.accuracy = None
        self.loss = None

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

    def create_model(self, base_model):
        """Creates the model from specified pretrained base model without top layer.
        Add two fully connected layers to train with dataset on custom classes

        Args:
            base_model (String): Model to load from tf.keras.applications

        Returns:
            Keras Model: Complete model to be trained
        """
        if base_model == "resnet50":
            base_model = ResNet50(input_shape=(900, 900, 3), weights='imagenet', include_top=False)
            preprocess_input = preprocess_resnet50
        elif base_model == "mobilenetv2":
            base_model = MobileNetV2(input_shape=(900, 900, 3), weights='imagenet', include_top=False)
            preprocess_input = preprocess_mobilenetv2
        elif base_model == "inceptionresnetv2":
            base_model = InceptionResNetV2(input_shape=(900, 900, 3), weights='imagenet', include_top=False)
            preprocess_input = preprocess_inceptionresnetv2
        else:
            print("Model string wrong. Use one of these models:\nresnet50, mobilenetv2, inceptionresnetv2")

        inputs = tf.keras.Input(shape=(900, 900, 3))
        x = preprocess_input(inputs)
        x = base_model(x, training=False)
        x = GlobalAveragePooling2D()(x)
        # x = Dropout(0.2)(x)
        # x = Dense(1024, activation='relu')(x)
        outputs = Dense(4, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs)

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

    def train(self, learning_rate):
        """Compile the model and train on dataset with specified learning rate.

        Args:
            learning_rate (float): Learning rate to apply
        """

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])

        history = self.model.fit(self.train_dataset,
                            validation_data=self.val_dataset,
                            epochs=5)

        self.accuracy = history.history['accuracy']
        self.loss = history.history['loss']

    def fine_tuning(self, learning_rate):
        self.model.trainable = True
        self.train(learning_rate)

    def plot_metrics(self):
        """Plot loss and accuracy over epochs
        """
        plt.figure(figsize=(8, 8))
        plt.plot(self.accuracy, label='Training Accuracy', marker="x")
        plt.plot(self.loss, label='Training Loss', marker="o")
        plt.legend(loc='lower left')
        plt.ylabel('Accuracy')
        plt.ylim([min(plt.ylim()), 1])
        plt.title('Training Accuracy and Loss')
        plt.show()

    def save_model(self, model_name):
        """Wrapper for Keras Model save function

        Args:
            model_name (String): Name to save model.h5 file
        """
        path = os.path.join("models", model_name + "_acc" + str(round(self.accuracy*100)) + ".h5")
        self.model.save(path, save_format='h5')

    def predict(self, image_path):
        """Wrapper for Keras Model predict function

        Args:
            image_path (Sring): Path to image to predict

        Returns:
            prediction(np.array): array of probabilities for each class
            class_index(int): argmax of prediction array
            class_name(String): name of class index
            image(cv2.Mat): predicted imagfe to display
        """
        image = cv2.imread(image_path)
        tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        tensor = tf.expand_dims(tensor, 0)
        prediction = self.model.predict(tensor)
        class_index = int(tf.math.argmax(prediction,1))
        class_name = self.train_dataset.class_names[class_index]

        return prediction, class_index, class_name, image

    def evaluate(self):
        """Test model on evaluation dataset

        Returns:
            dict: dict with two keys: accuracy, loss
        """
        result = self.model.evaluate(self.val_dataset, return_dict=True)
        self.accuracy = result['accuracy']
        self.loss = result['loss']

        return result


def main():
    path = os.path.join(os.getcwd(), "dataset", "augmented_dataset", "bottle", "images")

    model = TransferLearning(base_model="resnet50",
                             image_path=path,
                             image_size=(900, 900),
                             batch_size=16,
                             only_cpu=True,
                             model_path="models/resnet50.h5")

    # Best learning rates:
    # mobilenetv2 = 0.0005
    # resnet50 = 0.001

    # model.show_example_images()
    # model.train(learning_rate=0.0005)
    model.evaluate()
    model.plot_metrics()
    model.save_model(model_name="resnet50")

    prediction, class_index, class_name, image = model.predict(os.path.join(path, "good", "000.png"))
    print(prediction)
    cv2.imshow("predicted_class="+class_name, image)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
