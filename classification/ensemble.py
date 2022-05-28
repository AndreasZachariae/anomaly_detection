import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.svm import SVC
import pickle
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score

from classification.transfer_learning import TransferLearning
from classification.bag_of_visual_words import BagOfVisualWords


class Ensemble():
    def __init__(self, models_path, models, type_name="bottle", load_predinfo_name=None, load_meta_name=None):
        self.type_name = type_name
        self.model_list = list()
        for model in models:
            if "transfer_learning" in model[0]:
                self.model_list.append(
                    TransferLearning(
                        type_name=self.type_name,
                        model_path=os.path.join(models_path, model[0], self.type_name),
                        load_model_name=model[1],
                        only_cpu=False))
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
        self.accuracies = dict()

    def load_meta(self, meta_name):
        """
        Loads the meta model from a file.

        Parameters
        ----------
        meta_name : str
            The filename's prefix.

        Returns
        -------
        None.

        """
        self.meta_model = joblib.load(os.path.join("models", "ensemble", self.type_name, "meta_learners", f"{meta_name}_meta.joblib"))
        print("Pretrained meta learner loaded.")

    def load_prediction_info(self, predinfo_name):
        """
        Loads the model order and dict.

        Parameters
        ----------
        predinfo_name : str
            The filenames' prefix.

        Returns
        -------
        None.

        """
        pred_path = os.path.join(os.getcwd(), "models", "ensemble", self.type_name, "preds_data")
        self.class_order = np.load(
            os.path.join(pred_path, f"{predinfo_name}_order.npy"),
            allow_pickle=True)
        with open(os.path.join(pred_path, f"{predinfo_name}_classes.pkl"), "rb") as dict_file:
            self.class_dict = pickle.load(dict_file)

    def load_predictions(self, pred_name):
        """
        Load the prediction files.

        Parameters
        ----------
        pred_name : str
            The filenames' prefix.

        Returns
        -------
        predictions : numpy.ndarray
            Contains the predicted class probabilities for every model.
        labels : numpy.ndarray
            Contains the ground truth class indices.

        """
        target_path = os.path.join(os.getcwd(), "models", "ensemble", self.type_name, "preds_data")
        predictions = np.load(
            os.path.join(target_path, f"{pred_name}_features.npy"),
            allow_pickle=True)
        labels = np.load(
            os.path.join(target_path, f"{pred_name}_labels.npy"),
            allow_pickle=True)
        return (predictions, labels)

    def save_meta(self, meta_model, meta_name):
        """
        Saves the trained meta learner.

        Parameters
        ----------
        meta_model : sklearn.svm.SVC
            The meta learner to be saved.
        meta_name : str
            Prefix for the filename.

        Returns
        -------
        None.

        """
        target_path = os.path.join(os.getcwd(), "models", "ensemble", self.type_name, "meta_learners")
        joblib.dump(meta_model, os.path.join(target_path, f"{meta_name}_meta.joblib"))
        print(f"Saved meta learner in {target_path}.")

    def save_predictions(self, predictions, labels, order, dictionary, pred_name):
        """
        Saves all infos that will be necessary for reloading the level 0 predictions.

        Parameters
        ----------
        predictions : numpy.ndarray
            Contains the predicted class probabilities for every model.
        labels : numpy.ndarray
            Contains the ground truth class indices.
        order : list
            Contains the level 0 classifiers' orders of classes corresponding the meta learners order
        dictionary : dict
            This ensemble's dict to assign the class indices to the class names.
        pred_name : str
            Prefix for the filenames.

        Returns
        -------
        None.

        """
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
        """
        Implements the level 0 classification with all the ensemble's models.

        Parameters
        ----------
        images_path : str
            Path to the training images.
        save_name : str, optional
            Name for the predictions to be saved as if desired.

        Returns
        -------
        predictions : numpy.ndarray
            Contains the predicted class probabilities for every model.
        labels : numpa.ndarray
            Contains the ground truth class indices.

        """
        print("Calculating level 0 predictions...")
        class_names = os.listdir(images_path)

        self.set_class_dict(class_names)
        self.set_class_order(len(class_names), self.class_dict)

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
        """
        Trains the meta learner.

        Parameters
        ----------
        features : numpy.ndarray
            The training input containing the level 0 classification class probabilities.
        labels : numpy.ndarray
            The ground truth class indices.
        kernel_function : str, optional
            Kerne function for the meta learner (support vector machine). The default is "rbf".
        max_iter : int, optional
            Maximal iterations for the meta learner. No limit when set to -1. The default is 80000.
        eval_paths : TYPE, optional
            Paths for the evaluation of the meta learner if evaluation is desired. The default is (None, None, None).
        save_name : TYPE, optional
            Filename for the meta learner to be saved as if desired. The default is None.

        Returns
        -------
        None.

        """
        print("Training meta model...")
        self.meta_model = SVC(kernel=kernel_function, max_iter=max_iter, probability=True)
        self.meta_model.fit(features, labels)

        acc = [-1]
        if eval_paths[0] is not None or eval_paths[1] is not None:
            print("Evaluating...")
            self.evaluate_meta(eval_paths[0], eval_paths[1], eval_paths[2])
            acc = [self.metrics["accuracy"]["meta learner"]]

        if save_name is not None:
            if acc[-1] > 0:
                save_name += f"_acc{round(acc[-1]*100)}"
            self.save_meta(self.meta_model, save_name)
        print("Training done.")

    def predict(self, image_path, meta=False, weights=None, soft=False):
        """
        Implements level 1 classification for a single image.

        Parameters
        ----------
        image_path : str
            Path to an image (including the image file's name).
        meta : bool, optional
            Determines whether the classification is desired to be performed by the meta learner. The default is False.
        weights : list, optional
            A list of weights only being passed when weighted voting is desired.
        soft : bool, optional
            Determines whether soft voting is desired. Otherwise, hard voting will be performed. The default is False.

        Returns
        -------
        pred_class_idx : int
            The predicted class index.
        pred_class_name : str
            The predicted class name.

        """
        predictions = list()

        for i, model in enumerate(self.model_list):
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
        """
        Implements hard voting, soft voting and weighted voting.

        Parameters
        ----------
        probs : numpy.ndarray
            Contains the probabilities for each class predicted by all level 0 classfiers.
        weights : list, optional
            A list of weights only being passed when weighted voting is desired.
        soft : bool, optional
            Determines whether soft voting is desired. Otherwise, hard voting will be performed. The default is False.

        Returns
        -------
        majority_idx : int
            The resulting predicted class (level 1 prediction).

        """
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
        """
        Calculates accuracy, precision, recall, f1 score and confusion matrix for the meta learner and all implemented voting types.

        Returns
        -------
        predictions : dict
            The predicted classes sorted by the prediction method (meta learner, voting types).

        """
        predictions = dict()
        meta_preds = self.evaluate_meta(features_labels=(features, labels))
        predictions["meta learner"] = meta_preds

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

        self.metrics["accuracy"]["hard voting"] = accuracy_score(labels, hard_preds)
        self.metrics["confusion matrix"]["hard voting"] = confusion_matrix(labels, hard_preds)
        self.metrics["precision"]["hard voting"] = precision_score(labels, hard_preds, average="weighted")
        self.metrics["recall"]["hard voting"] = recall_score(labels, hard_preds, average="weighted")
        self.metrics["f1"]["hard voting"] = f1_score(labels, hard_preds, average="weighted")
        predictions["hard voting"] = hard_preds

        self.metrics["accuracy"]["soft voting"] = accuracy_score(labels, soft_preds)
        self.metrics["confusion matrix"]["soft voting"] = confusion_matrix(labels, soft_preds)
        self.metrics["precision"]["soft voting"] = precision_score(labels, soft_preds, average="weighted")
        self.metrics["recall"]["soft voting"] = recall_score(labels, soft_preds, average="weighted")
        self.metrics["f1"]["soft voting"] = f1_score(labels, soft_preds, average="weighted")
        predictions["soft voting"] = soft_preds

        self.metrics["accuracy"]["weighted voting"] = accuracy_score(labels, weighted_preds)
        self.metrics["confusion matrix"]["weighted voting"] = confusion_matrix(labels, weighted_preds)
        self.metrics["precision"]["weighted voting"] = precision_score(labels, weighted_preds, average="weighted")
        self.metrics["recall"]["weighted voting"] = recall_score(labels, weighted_preds, average="weighted")
        self.metrics["f1"]["weighted voting"] = f1_score(labels, weighted_preds, average="weighted")
        predictions["weighted voting"] = weighted_preds

        return predictions

    def evaluate_meta(self, load_pred_name=None, pred_images_path=None, pred_save_name=None, features_labels=None):
        """
        Calculates accuracy, precision, recall, f1 score and confusion matrix of the meta learner.

        Returns
        -------
        predictions : numpy.ndarray
            The classes predicted by the  meta learner for the evaluation.

        """
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

        if "accuracy" in self.metrics:
            self.metrics["accuracy"]["meta learner"] = accuracy_score(labels, predictions)
        else:
            new_dict = dict()
            new_dict["meta learner"] = accuracy_score(labels, predictions)
            self.metrics["accuracy"] = new_dict

        if "confusion matrix" in self.metrics:
            self.metrics["confusion matrix"]["meta learner"] = confusion_matrix(labels, predictions)
        else:
            new_dict = dict()
            new_dict["meta learner"] = confusion_matrix(labels, predictions)
            self.metrics["confusion matrix"] = new_dict

        if "precision" in self.metrics:
            self.metrics["precision"]["meta learner"] = precision_score(labels, predictions, average="weighted")
        else:
            new_dict = dict()
            new_dict["meta learner"] = precision_score(labels, predictions, average="weighted")
            self.metrics["precision"] = new_dict

        if "recall" in self.metrics:
            self.metrics["recall"]["meta learner"] = recall_score(labels, predictions, average="weighted")
        else:
            new_dict = dict()
            new_dict["meta learner"] = recall_score(labels, predictions, average="weighted")
            self.metrics["recall"] = new_dict

        if "f1" in self.metrics:
            self.metrics["f1"]["meta learner"] = f1_score(labels, predictions, average="weighted")
        else:
            new_dict = dict()
            new_dict["meta learner"] = f1_score(labels, predictions, average="weighted")
            self.metrics["f1"] = new_dict

        return predictions

    def set_class_dict(self, class_names):
        """
        Determines the class dict based on the given class names.

        Parameters
        ----------
        class_names : list
            Contains the class names.

        Returns
        -------
        None.

        """
        class_indices = list(range(len(class_names)))
        self.class_dict = dict(zip(class_names, class_indices))
        self.class_dict.update(dict(zip(class_indices, class_names)))

    def set_class_order(self, nr_of_classes, class_dict):
        """
        Calculates the level 0 classifiers' orders of classes corresponding the meta learners order.

        Parameters
        ----------
        nr_of_classes : int
            The amount of classes.
        class_dict : dict
            This ensemble's dict to assign the class indices to the class names.

        Returns
        -------
        None.

        """
        self.class_order = list()
        for model in self.model_list:
            try:
                translator = model.labels
            except AttributeError:
                translator = model.class_dict
            order = list()
            for i in range(nr_of_classes):
                order.append(class_dict[translator[i]])
            self.class_order.append(order)

    def plot_confusion_matrix(self, y_true, y_pred, filepath):
        """
        Plots the confusion matrix resulting from true and predicted class indices.

        Parameters
        ----------
        y_true : numpy.ndarray
            The ground truth class indices.
        y_pred : numpy.ndarray
            The predicted class indices.
        filepath : str
            Path including the filename for the plot to be stored.

        Returns
        -------
        None.

        """
        labels = list()
        for i in range(int(len(self.class_dict)/2)):
            labels.append(self.class_dict[i])
        disp = ConfusionMatrixDisplay.from_predictions(y_true,
                                                       y_pred,
                                                       display_labels=labels,
                                                       cmap=plt.cm.Blues,
                                                       normalize="all",
                                                       xticks_rotation="vertical")
        disp.ax_.set_title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(filepath)

    def plot_confusion_matrix_old_tf_gpu(self, y_true, y_pred, filepath):
        """
        Plots the confusion matrix resulting from true and predicted class indices.
        The sklearn function ConfusionMatrixDisplay.from_predictions was introduced in v1.0.0
        This method uses the old function from sklearn 0.24 with tensorflow-gpu 2.3 on python 3.6

        Parameters
        ----------
        y_true : numpy.ndarray
            The ground truth class indices.
        y_pred : numpy.ndarray
            The predicted class indices.
        filepath : str
            Path including the filename for the plot to be stored.

        Returns
        -------
        None.

        """
        labels = list()
        for i in range(int(len(self.class_dict)/2)):
            labels.append(self.class_dict[i])

        cf_matrix = confusion_matrix(y_true, y_pred, normalize="all")
        disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix,
                                      display_labels=labels)

        disp.plot(cmap=plt.cm.Blues)
        plt.savefig(filepath)


def main():
    load_all = True
    load_data = False

    type_name = "hazelnut"
    # weights = [0.74, 0.88, 0.83, 0.60]  # accuracies from bottle models
    weights = [0.91, 0.92, 0.92, 0.71]  # accuracies from hazelnut models

    models_path = os.path.join(os.getcwd(), "models")
    # first colulmn needs to be bovw for BagOfVisualWords and contain transfer_learning for TransferLearning
    # models = [["transfer_learning_old", "inceptionresnetv2.h5"],
    #           ["transfer_learning_old", "mobilenetv2.h5"],
    #           ["transfer_learning_old", "resnet50.h5"],
    #           ["bovw", "bottle"]]
    # models = [["transfer_learning", "inceptionresnetv2_bottle_acc74.h5"],
    #           ["transfer_learning", "mobilenetv2_bottle_acc88.h5"],
    #           ["transfer_learning", "resnet50_bottle_acc83.h5"],
    #           ["bovw", "bottle"]]
    models = [["transfer_learning", "inceptionresnetv2_hazelnut_acc91.h5"],
              ["transfer_learning", "mobilenetv2_hazelnut_acc92.h5"],
              ["transfer_learning", "resnet50_hazelnut_acc92.h5"],
              ["bovw", "hazelnut"]]
    images_path = os.path.join(os.getcwd(), "dataset", "augmented_dataset", type_name, "test", "images")

    if load_all:
        ensemble = Ensemble(models_path=models_path, models=models, type_name=type_name, load_predinfo_name="val_new", load_meta_name="sigmoid_new_acc94.0")
        ensemble.load_prediction_info("test_new")
        features, labels = ensemble.load_predictions("test_new")
        predictions = ensemble.evaluate(features, labels, weights=weights)
        print(ensemble.class_dict)
        for metric in ensemble.metrics:
            if not metric == "confusion matrix":
                print(metric + ":")
            for voting_type in ensemble.metrics[metric]:
                if metric == "confusion matrix":
                    filepath = os.path.join(models_path, "ensemble", type_name, voting_type.split()[0]+"_confusion_matrix")
                    ensemble.plot_confusion_matrix_old_tf_gpu(labels, predictions[voting_type], filepath)
                else:
                    print(f"\t{voting_type}: {ensemble.metrics[metric][voting_type]}")

    elif load_data:
        ensemble = Ensemble(models_path=models_path, models=models, type_name=type_name, load_predinfo_name="val_new")
        predictions, labels = ensemble.load_predictions("val_new")
        ensemble.train(predictions,
                       labels,
                       kernel_function="sigmoid",
                       max_iter=-1,
                       eval_paths=("test_new", None, None),
                       save_name="sigmoid_new")
        print("Accuracy:", ensemble.metrics["accuracy"]["meta learner"])
        print("Confusion matrix:", ensemble.metrics["confusion matrix"]["meta learner"])

    else:
        ensemble = Ensemble(models_path=models_path, models=models, type_name=type_name)
        predictions, labels = ensemble.predict_level_0(images_path.replace("test", "validate"), save_name="val_new")
        ensemble.train(predictions,
                       labels,
                       kernel_function="sigmoid",
                       max_iter=-1,
                       eval_paths=(None, images_path, "test_new"),
                       save_name="sigmoid_new")
        print("Accuracy:", ensemble.metrics["accuracy"]["meta learner"])
        print("Confusion matrix:", ensemble.metrics["confusion matrix"]["meta learner"])

    img_path = os.path.join(images_path, "good", "156_train.png")
    print("--- Predictions for image", img_path)
    print("\tMeta:", ensemble.predict(img_path, meta=True))
    print("\tWeighted:", ensemble.predict(img_path, weights=weights))
    print("\tSoft voting:", ensemble.predict(img_path, soft=True))
    print("\tHard voting:", ensemble.predict(img_path))


if __name__ == '__main__':
    main()
