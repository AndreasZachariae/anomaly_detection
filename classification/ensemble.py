import os
from transfer_learning import TransferLearning

class Ensemble():
    def __init__(self, image_path, image_size, batch_size, only_cpu, model_path=None):
        if only_cpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            print("Disabled GPU devices")

        self.model_list = [TransferLearning("mobilenetv2", False, "models/mobilenetv2.h5"),
                           TransferLearning("resnet50", False, "models/resnet50.h5"),
                           TransferLearning("inceptionresnetv2", False, "models/inceptionresnetv2.h5")]

        self.meta_model = None
        self.train_dataset = None
        self.val_dataset = None
        self.accuracy = dict()
        self.loss = dict()

    def predict_all(self, image_path):
        predictions = []
        for model in self.model_list:
            prediction, class_index, _, _ = model.predict(image_path)
            predictions.append((class_index, prediction[0][class_index]))

        print("[Model1(predicted_class, probability), ...]")
        print(predictions)

def main():
    path = os.path.join(os.getcwd(), "dataset", "augmented_dataset", "bottle", "images")

    ensemble = Ensemble(image_path=path,
                     image_size=(900, 900),
                     batch_size=16,
                     only_cpu=True,
                     model_path="models/mobilenetv2.h5")

    ensemble.predict_all(os.path.join(path, "good", "000.png"))

if __name__ == '__main__':
    main()
    
        