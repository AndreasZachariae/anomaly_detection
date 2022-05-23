import os
import numpy as np
from collections import Counter

from classification.transfer_learning import TransferLearning
from classification.bag_of_visual_words import BagOfVisualWords

class Ensemble():
    def __init__(self, model_path, models, object_type="bottle"):    
        self.model_list = list()
        
        for model in models:
            if model[0] == "transfer_learning":
                # TODO: remove _old
                self.model_list.append(TransferLearning(type_name=object_type,
                                                        model_path=os.path.join(model_path, f"{model[0]}_old", object_type),
                                                        load_model_name=model[1],
                                                        only_cpu=True))
            elif model[0] == "bovw":
                self.model_list.append(BagOfVisualWords(os.path.join(model_path, model[0], model[1])))

        # TODO: create SVM
        self.meta_model = None

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
            majority_idx = np.argmax(np.sum(weighted_probs, axis=0))
        else:
            if soft:
                majority_idx = np.argmax(np.sum(probs, axis=0))
            else:
                cnt = Counter(predictions[:,0])
                majority_idx = cnt.most_common(1)[0][0] 
                
        majority_name = predictions[np.where(predictions[:, 0] == majority_idx), 1][0][0]
                
        return (majority_idx, majority_name)

def main():
    model_path = os.path.join(os.getcwd(), "models")
    models = [["transfer_learning", "inceptionresnetv2.h5"],
              ["transfer_learning", "mobilenetv2.h5"],
              ["transfer_learning", "resnet50.h5"],
              ["bovw", "bottle"]]

    ensemble = Ensemble(model_path=model_path, models=models, object_type="bottle")

    path = os.path.join(os.getcwd(), "dataset", "augmented_dataset", "bottle", "test", "images")
    ensemble.predict_all(os.path.join(path, "broken_large", "004.png"))

if __name__ == '__main__':
    main()
    
        