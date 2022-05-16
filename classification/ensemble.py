import os
import numpy as np
from collections import Counter

from classification.transfer_learning import TransferLearning
from classification.bag_of_visual_words import BagOfVisualWords

class Ensemble():
    def __init__(self, image_path, image_size, batch_size, only_cpu, model_path=None):
        if only_cpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            print("Disabled GPU devices")

        self.model_list = [TransferLearning("mobilenetv2", False, os.path.join("models","mobilenetv2.h5")),
                           TransferLearning("resnet50", False, os.path.join("models","resnet50.h5")),
                           TransferLearning("inceptionresnetv2", False, os.path.join("models","inceptionresnetv2.h5")),
                           BagOfVisualWords(os.path.join("models","bovw"))]

        self.meta_model = None
        self.train_dataset = None
        self.val_dataset = None
        self.accuracy = dict()
        self.loss = dict()

    def predict_all(self, image_path):
        predictions = list()

        for model in self.model_list:
            prediction, class_index, class_name, _ = model.predict(image_path)
            predictions.append([class_index, class_name, prediction[0][class_index], list(prediction[0])])
            
        predictions = np.array(predictions, dtype=object)

        print("[Model1[predicted class index, predicted class name, probability],\n ...]")
        print(predictions[:,:-1])
        print("Hard voting (majority voting):", self.find_majority(predictions))
        print("Soft voting:", self.find_majority(predictions, soft=True))
        # TODO: change weights (depending on accuracy?)
        print("Weighted voting:", self.find_majority(predictions, weights=[0.2, 0.2, 0.2, 0.4]))
        
    def find_majority(self, predictions, weights=None, soft=False):
        probs = np.array(list(predictions[:,3]), dtype=float)
        
        if weights is not None:
            weighted_probs = np.array([weights]).T * probs
            majority = np.argmax(np.sum(weighted_probs, axis=0))
        else:
            if soft:
                majority = np.argmax(np.sum(probs, axis=0))
            else:
                cnt = Counter(predictions[:,0])
                majority = cnt.most_common(1)[0][0] 
                
        return majority

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
    
        