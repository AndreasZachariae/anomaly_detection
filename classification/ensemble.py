import os
import numpy as np
from collections import Counter
from sklearn.svm import SVC 
import pickle
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix

from classification.transfer_learning import TransferLearning
from classification.bag_of_visual_words import BagOfVisualWords

# TODO: function descriptions

class Ensemble():
    def __init__(self, models_path, models, type_name="bottle", load_predinfo_name=None, load_meta_name=None):
        self.type_name = type_name
        self.model_list = list()
        for model in models:
            # TODO: Change back to equals (==)
            if "transfer_learning" in model[0]:
                self.model_list.append(
                    TransferLearning(
                        type_name=self.type_name,
                        model_path=os.path.join(models_path, model[0], self.type_name),
                        load_model_name=model[1],
                        only_cpu=True))
            elif model[0] == "bovw":
                self.model_list.append(BagOfVisualWords(os.path.join(models_path, model[0], model[1])))
            else:
                print(f"Invalid type of model: {model[0]}. Has to be one of \"transfer_learning\" or \"bovw\".")
        
        self.meta_model = None
        self.class_dict = None
        self.class_order = None
        if load_meta_name is not None:
            self.load_meta(load_meta_name)
        if load_predinfo_name is not None:
            self.load_prediction_info(load_predinfo_name)  
        self.metrics = dict()
 
    def load_meta(self, meta_name):
        self.meta_model = joblib.load(os.path.join("models", "ensemble", self.type_name, "meta_learners", f"{meta_name}_meta.joblib"))
        print("Pretrained meta learner loaded.")
    
    def load_prediction_info(self, predinfo_name):
        pred_path = os.path.join(os.getcwd(), "models", "ensemble", self.type_name, "preds_data")
        self.class_order = np.load(
            os.path.join(pred_path, f"{predinfo_name}_order.npy"),
            allow_pickle=True)
        with open(os.path.join(pred_path, f"{predinfo_name}_classes.pkl"), "rb") as dict_file:
                self.class_dict = pickle.load(dict_file) 
    
    def load_predictions(self, pred_name):
        target_path = os.path.join(os.getcwd(), "models", "ensemble", self.type_name, "preds_data")
        predictions = np.load(
            os.path.join(target_path, f"{pred_name}_features.npy"),
            allow_pickle=True)
        labels = np.load(
            os.path.join(target_path, f"{pred_name}_labels.npy"),
            allow_pickle=True)
        return (predictions, labels)
    
    def save_meta(self, meta_model, meta_name):
        target_path = os.path.join(os.getcwd(), "models", "ensemble", self.type_name, "meta_learners")
        joblib.dump(meta_model, os.path.join(target_path, f"{meta_name}_meta.joblib")) 
        print(f"Saved meta learner in {target_path}.")
        
    def save_predictions(self, predictions, labels, order, dictionary, pred_name):
        target_path = os.path.join(os.getcwd(), "models", "ensemble", self.type_name, "preds_data")
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        np.save(
            os.path.join(target_path, f"{pred_name}_features.npy"),
            predictions, allow_pickle=True)
        np.save(
            os.path.join(target_path, f"{pred_name}_labels.npy"),
            labels, allow_pickle=True)
        np.save(
            os.path.join(target_path, f"{pred_name}_order.npy"),
            order, allow_pickle=True)
        with open(os.path.join(target_path, f"{pred_name}_classes.pkl"), "wb") as dict_file:
            pickle.dump(dictionary, dict_file)
        print(f"Saved predictions and info in {target_path}.")

    def predict_level_0(self, images_path, save_name=None):
        print("Calculating level 0 predictions...") 
        class_names = os.listdir(images_path)        

        self.set_class_dict(class_names)
        self.set_class_order(class_names, self.class_dict)
        
        predictions = list()
        labels = list()
        
        for class_name in class_names:
            print(f"Predictions for \"{class_name}\"...")
            c_path = os.path.join(images_path, class_name)
            label = self.class_dict[class_name]
            for img in os.listdir(c_path):
                img = os.path.join(c_path, img)
                labels.append(label)
                proba_list = list()
                for idx, model in enumerate(self.model_list):
                    model_class_order = self.class_order[idx]
                    probas, _, _, _ = model.predict(img)
                    proba_list += list(np.array(probas[0])[model_class_order])
                predictions.append(proba_list)
        
        print("Level 0 predictions done.")
        
        if save_name is not None:
            self.save_predictions(predictions, labels, self.class_order, self.class_dict, save_name)
            
        return (predictions, labels)    
    
    def train(self, features, labels, kernel_function="rbf", max_iter=80000, eval_paths=(None, None, None), save_name=None):
        print("Training meta model...")
        self.meta_model = SVC(kernel=kernel_function, max_iter=max_iter, probability=True)
        self.meta_model.fit(features, labels)
        
        acc = [-1]
        if eval_paths[0] is not None or eval_paths[1] is not None:
            print("Evaluating...")
            acc = [self.evaluate_meta(eval_paths[0], eval_paths[1], eval_paths[2])]
            
        if save_name is not None:
            if acc[-1] > 0:
                save_name += f"_acc{round(acc[-1]*100)}"
            self.save_meta(self.meta_model, save_name)
        print("Training done.")
        
    def predict(self, image_path, meta=False, weights=None, soft=False):
        predictions = list()

        for i,model in enumerate(self.model_list):
            prediction, _, _, _ = model.predict(image_path)
            predictions.append(list(prediction[0][self.class_order[i]]))
            
        probs = np.array(predictions, dtype=float)
        
        if meta:
            pred_class_idx = self.meta_model.predict([probs.flatten()])[0]
        else:
            pred_class_idx = self.get_majority(probs, weights, soft)
            
        pred_class_name = self.class_dict[pred_class_idx]
        
        return (pred_class_idx, pred_class_name)
    
    def get_majority(self, probs, weights=None, soft=False):
        # TODO: what to do with a tie? in voting, currently: the first occurance
        if weights is not None:
            weighted_probs = np.array([weights]).T * probs
            majority_idx = np.argmax(np.sum(weighted_probs, axis=0))
        else:
            if soft:
                majority_idx = np.argmax(np.sum(probs, axis=0))
            else:
                cnt = Counter(np.argmax(probs, axis=1))
                majority_idx = cnt.most_common(1)[0][0] 
                
        return majority_idx
    
    def evaluate(self, features, labels, weights, confusion_labels=None):
        if not "meta learner" in self.metrics:
            self.evaluate_meta(features_labels=(features, labels))
            
        num_models = len(self.model_list)
        num_classes = int(len(features[0])/num_models)
        probs = np.reshape(features, (-1, num_models, num_classes))
        
        hard_preds = list()
        soft_preds = list()
        weighted_preds = list()
        for img_proba in probs:
            hard_preds.append(self.get_majority(img_proba))
            soft_preds.append(self.get_majority(img_proba, soft=True))
            weighted_preds.append(self.get_majority(img_proba, weights=weights))    
        
        self.metrics["hard voting"] = (accuracy_score(labels, hard_preds), confusion_matrix(labels, hard_preds))
        self.metrics["soft voting"] = (accuracy_score(labels, soft_preds), confusion_matrix(labels, soft_preds))
        self.metrics["weighted voting"] = (accuracy_score(labels, weighted_preds), confusion_matrix(labels, weighted_preds))
            
        
    def evaluate_meta(self, load_pred_name=None, pred_images_path=None, pred_save_name=None, features_labels=None):
        if features_labels is not None:
            features = features_labels[0]
            labels = features_labels[1]
        elif load_pred_name is not None:
            self.load_prediction_info(load_pred_name)
            features, labels = self.load_predictions(load_pred_name)
        elif pred_images_path is not None:
            features, labels = self.predict_level_0(pred_images_path, pred_save_name)
        else:
            print("Invalid parameters. Either load_pred_name, pred_images_path or features_labels has to be specified.")
            
        predictions = self.meta_model.predict(features)
        self.metrics["meta learner"] = (accuracy_score(labels, predictions), confusion_matrix(labels, predictions))
        
        return self.metrics["meta learner"]
        
    def set_class_dict(self, class_names):       
        class_indices = list(range(len(class_names)))
        self.class_dict = dict(zip(class_names, class_indices))
        self.class_dict.update(dict(zip(class_indices, class_names)))
     
    def set_class_order(self, class_names, class_dict):       
        self.class_order = list()
        for model in self.model_list:
            try:
                translator = model.labels
            except AttributeError:
                translator = model.class_dict
            order = list()
            for i in range(len(class_names)):
                order.append(class_dict[translator[i]])
            self.class_order.append(order)
        

def main():
    load_all = True
    load_data = False
    
    models_path = os.path.join(os.getcwd(), "models")
    models = [["transfer_learning_old", "inceptionresnetv2.h5"],
              ["transfer_learning_old", "mobilenetv2.h5"],
              ["transfer_learning_old", "resnet50.h5"],
              ["bovw", "bottle"]]
    # models = [["transfer_learning", "inceptionresnetv2_bottle_acc87.h5"],
    #           ["transfer_learning", "mobilenetv2_bottle_acc88.h5"],
    #           ["transfer_learning", "resnet50_bottle_acc84.h5"],
    #           ["bovw", "bottle"]]
    images_path = os.path.join(os.getcwd(), "dataset", "augmented_dataset", "bottle", "test", "images")

    
    if load_all:
        ensemble = Ensemble(models_path=models_path, models=models, type_name="bottle", load_predinfo_name="val", load_meta_name="sigmoid_bottle_max-1_acc83")
        # print("Accuracy meta learner:", ensemble.evaluate_meta("test"))
        ensemble.load_prediction_info("test")
        features, labels = ensemble.load_predictions("test")
        ensemble.evaluate(features, labels, weights=[0.87, 0.88, 0.84, 0.61])
        print(ensemble.class_dict)
        for key in ensemble.metrics:
            print(f"--- Accuracy and confusion matrix {key}:\n{ensemble.metrics[key][0]}\n{ensemble.metrics[key][1]}")
        # Accuracies with old transfer_learning models (but accuracies of new ones as weights...), new bovw on test dataset:
        # Meta learner: 0.8324697329842932
        # Hard voting: 0.774869109947644
        # Soft voting: 0.7382198952879581
        # Weighted voting: 0.6701570680628273
        
    elif load_data:
        ensemble = Ensemble(models_path=models_path, models=models, type_name="bottle", load_predinfo_name="val")
        predictions, labels = ensemble.load_predictions("val")
        ensemble.train(predictions, labels, kernel_function="sigmoid", max_iter=-1, eval_paths=("test",None,None), save_name="sigmoid_bottle_max-1")
        print("Accuracy:", ensemble.metrics["meta learner"][0])
        print("Confusion matrix:", ensemble.metrics["meta learner"][1])

    else:
        ensemble = Ensemble(models_path=models_path, models=models, type_name="bottle")
        predictions, labels = ensemble.predict_level_0(images_path.replace("test", "validate"), save_name="val_new")
        ensemble.train(
            predictions, 
            labels, 
            kernel_function="sigmoid", 
            max_iter=-1, 
            eval_paths=(None, images_path, "test_new"), 
            save_name="sigmoid_new")
        print("Accuracy:", ensemble.metrics["meta learner"][0])
        print("Confusion matrix:", ensemble.metrics["meta learner"][1])

        
    
    img_path = os.path.join(images_path, "good", "019_train.png")
    print("--- Predictions for image", img_path)
    print("\tMeta:", ensemble.predict(img_path, meta=True))
    print("\tWeighted:", ensemble.predict(img_path, weights=[0.87, 0.88, 0.84, 0.61]))
    print("\tSoft voting:", ensemble.predict(img_path, soft=True))
    print("\tHard voting:", ensemble.predict(img_path))
        
        
    
if __name__ == '__main__':
    main()