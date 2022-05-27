# -*- coding: utf-8 -*-
import os
import collections
import joblib
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC 
from scipy.cluster.vq import kmeans,vq, whiten
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score


# CLASS -----------------------------------------------------------------------
class BagOfVisualWords(): 
    def __init__(self, model_path=None, descriptor_type="sift"):
        self.descriptor = self.get_descriptor(descriptor_type)
        self.codebook, self.svm, self.class_dict = self.get_model(model_path) 
        self.train_dataset = None
        self.val_dataset = None
        self.metrics = dict()
        
    # CLASSIFICATION FUNCTIONS ------------------------------------------------
    def predict(self, image_path):
        """
        Predict a specific image's class.

        Parameters
        ----------
        image_path : str
            Path to the image.

        Returns
        -------
        prediction : numpy.ndarray
            Array of probabilities for each class.
        class_index : int
            Argmax of prediction array.
        class_name : str
            The corresponding class name to the class index.
        image : numpy.ndarray
            The classified image.

        """
        (img_descr, _) = self.get_descriptor_lists(np.array([image_path]), self.descriptor)
        features = self.get_codewords(img_descr)
        
        prediction = self.svm.predict_proba(features)
        class_index = int(np.argmax(prediction))
        class_name = self.class_dict[class_index]
        image = cv2.imread(image_path)
        
        return prediction, class_index, class_name, image
    
    def train(self, svm_type="rbf", svm_iter=80000, k=200, k_iter=20):
        """
        Train a Support Vector Machine (SVM) with Bag of Visual Words.

        Parameters
        ----------
        svm_type : str, optional
            The SVM's kernel type. The default is "rbf".
        svm_iter : int, optional
            A limit of iterations for the SVM. -1 for no limit. The default is 80000.
        k : int, optional
            The number of centroids for k-means clustering. The default is 200.
        k_iter : int, optional
            The number of times to run k-means. The default is 20.

        Returns
        -------
        None.

        """
        print("\nTraining started...")
        
        try:
            self.svm = SVC(kernel=svm_type, max_iter=svm_iter, probability=True)
        except ValueError:
            print("Invalid parameter for the Support Vector Machine. svm_type"
                  + " must be one of \"linear\", \"poly\", \"rbf\", \"sigmoid"
                  + ", or \"precomputed\". svm_iter must be a positive integer"
                  + " or -1 for no limit.")
            return
        
        print("Extract keypoint features...")
        (img_descr, descrs) = self.get_descriptor_lists(self.train_dataset[0], self.descriptor)
        print("Create codebook...")
        # rescale to give features unit variance before clustering
        descrs = whiten(descrs)
        self.codebook, _ = kmeans(descrs, k, k_iter) 
        print("Create bag of codewords...")
        features = self.get_codewords(img_descr)
        
        print("Train Support Vector Machine...")
        self.svm.fit(features, np.array(self.train_dataset[1]))
                
        self.evaluate()
        
        print("Training done.")
        
    def evaluate(self):
        """
        Calculates accuracy, precision, recall, f1 score and confusion matrix.

        Returns
        -------
        predictions : numpy.ndarray
            The classes predicted by the trained SVM for the validation dataset.

        """
        print("Calculating metrics...") 
        print("Extract keypoint features...")    
        (img_descr, _) = self.get_descriptor_lists(self.val_dataset[0], self.descriptor)
        print("Create bag of codewords...")
        features = self.get_codewords(img_descr)
        
        print("Predict data with Support Vector Machine...")
        predictions = self.svm.predict(features)
        
        self.metrics["accuracy"] = [accuracy_score(self.val_dataset[1], predictions)]
        self.metrics["confusion matrix"] = confusion_matrix(self.val_dataset[1], predictions)
        self.metrics["precision"] = precision_score(self.val_dataset[1], predictions, average="weighted")
        self.metrics["recall"] = recall_score(self.val_dataset[1], predictions, average="weighted")
        self.metrics["f1"] = f1_score(self.val_dataset[1], predictions, average="weighted")
        
        print("Calculating done.")
        
        return predictions
    
    def get_codewords(self, img_descr_list):
        """
        Creates the Bag of Visual Words.

        Parameters
        ----------
        img_descr_list : numpy.ndarray
            A matrix containing the path and corresponding feature descriptor for every image.

        Returns
        -------
        features : numpy.ndarray
            A matrix containing the number of occurences of every word for every image.

        """
        features = np.zeros((len(img_descr_list), self.codebook.shape[0]), dtype=float)
        for i, (_, des) in enumerate(img_descr_list):
            # assign features from the descriptor to words from the codebook
            words, _ = vq(des, self.codebook)
            # count the occurences of the codebook's words
            for w in words:
                features[i][w] += 1
        
        return features
    
    def get_descriptor_lists(self, img_paths, descriptor):
        """
        Calculates feature descriptors for every image and stores them in two different ways.

        Parameters
        ----------
        img_paths : numpy.ndarray
            An array containing all image paths.
        descriptor : cv2.Feature2D (specifically cv2.ORB or cv2.SIFT)
            The keypoint detector and descriptor extractor object.

        Returns
        -------
        img_des : numpy.ndarray
            A matrix containing the path and corresponding feature descriptor for every image.
        descriptors : numpy.ndarray
            An array containing the feature descriptor for every image.

        """
        img_des = list()
        descriptors = list(list())
        
        for path in img_paths:
            img = cv2.imread(path)
            # detect keypoints and compute descriptor
            _, dscr = descriptor.detectAndCompute(img, None)
            img_des.append([path, dscr])
            descriptors += list(dscr)
           
        img_des = np.array(img_des, dtype=object)
        descriptors = np.array(descriptors, dtype=float)
           
        return (img_des, descriptors)
        
    # DATA PROCESSING FUNCTIONS -----------------------------------------------
    # def load_data(self, image_path, image_size=None, batch_size=None, validation_split=0.2):
    #     """
    #     Loads the data at a specific path and randomly splits it into training and validation datasets.

    #     Parameters
    #     ----------
    #     image_path : str
    #         Path to the images.
    #     validation_split : float, optional
    #         The ratio of validation data to validation + training data. The default is 0.2.

    #     Returns
    #     -------
    #     None.

    #     """
    #     self.class_dict = self.get_classes(image_path)
    #     dataset = self.load_all(image_path) # dataset = (classes, labels)
    #     self.train_dataset, self.val_dataset = self.random_split(
    #         dataset[0], dataset[1], validation_split) 
        
    def load_data(self, image_path):
        """
        Loads the training and validation dataset.

        Parameters
        ----------
        image_path : str
            Parent path to the images for the datasets.

        Returns
        -------
        None.

        """
        train_path = os.path.join(image_path, "train", "images")
        self.class_dict = self.get_classes(train_path)
        self.train_dataset = self.shuffle_data(self.load_all(train_path))
        self.val_dataset = self.load_all(train_path.replace("train", "test"), is_test=True)
        
    def shuffle_data(self, dataset):
        """
        Shuffles the images and their class labels.

        Parameters
        ----------
        dataset : tuple
            Contains the images and their class labels.

        Returns
        -------
        Tuple of shuffled images and their class labels.

        """
        p = np.random.permutation(len(dataset[1]))
        images = dataset[0]
        labels = dataset[1]
        return (images[p], labels[p])
        
    # def random_split(self, images, labels, split_ratio):
    #     """
    #     Randomly splits a given dataset into training and test data.

    #     Parameters
    #     ----------
    #     images : numpy.ndarray
    #         An array containing all image paths.
    #     labels : numpy.ndarray
    #         An array containing the class labels corresponding to the images.
    #     split_ratio : float
    #         The ratio of validation data to validation + training data.

    #     Returns
    #     -------
    #     imgs_train : numpy.ndarray
    #         An array containing all training image paths.
    #     lbls_train : numpy.ndarray
    #         An array containing all training class labels.
    #     imgs_test : numpy.ndarray
    #         An array containing all test image paths.
    #     lbls_test : numpy.ndarray
    #         An array containing all test class labels.

    #     """
    #     # shuffle image paths and their class labels in the same way
    #     p = np.random.permutation(len(images))
    #     images = images[p]
    #     labels = labels[p]
        
    #     # split into training and test data
    #     split_value = int(len(images)*split_ratio)
    #     imgs_train = images[split_value:]
    #     lbls_train = labels[split_value:]
    #     imgs_val = images[:split_value]
    #     lbls_val = labels[:split_value]
        
    #     return (imgs_train, lbls_train), (imgs_val, lbls_val)
        
    def load_all(self, path, is_test=False):
        """
        Make a list of all images' paths and their classes respectively.

        Parameters
        ----------
        path : str
            Path to the directory containing all images sorted by class names.
        is_test : bool
            Determines whether the ratio of images per class is checked and adjusted.

        Returns
        -------
        img_paths : numpy.ndarray
            An array containing all image paths.
        img_classes : numpy.ndarray
            An array containing the class labels corresponding to the images.

        """
        # get all image paths in the given path and their classes
        img_paths = np.array([])
        img_classes = np.array([])
        for class_name in os.listdir(path):
            c_path = os.path.join(path, class_name)
            img_c_paths = [os.path.join(c_path, img_name) for img_name in os.listdir(c_path)]
            img_paths = np.append(img_paths, img_c_paths)
            img_classes = np.append(img_classes, [self.class_dict[class_name]] * len(img_c_paths))
        
        # count the images per class
        occ = collections.Counter(img_classes)
        print(f"Found {len(img_classes)} images:")
        
        # make sure there is about the same amount of images per class
        values = np.array(list(occ.values()))
        var_coeff = np.std(values)/np.mean(values)
        for key in occ:
            print(f"- {occ[key]} images of class \"{self.class_dict[key]}\".")
        if not is_test:
            if var_coeff >= 0.1:
                max_imgs = min(values)
                for key in occ:
                    pos = np.where(img_classes == key)[0]
                    random_indices = np.random.choice(pos, pos.shape[0]-max_imgs, replace=False)
                    img_paths = np.delete(img_paths, random_indices)
                    img_classes = np.delete(img_classes, random_indices)
                print(f"Using only {max_imgs} images per class.")           
            
        return (img_paths, img_classes)
        
    # GENERAL FUNCTIONS -------------------------------------------------------
    def save_model(self, model_name="bovw"):
        """
        Save the codebook, SVM and class dictionary necessary for reusing a pretrained model.

        Parameters
        ----------
        model_name : str, optional
            Name of the folder to be created containing the model files. The default is "bovw".

        Returns
        -------
        None.

        """
        # create folder if it does not exist already
        if not os.path.exists(os.path.join(os.getcwd(), "models", "bovw", model_name)):
            os.makedirs(os.path.join(os.getcwd(), "models", "bovw", model_name))
        
        # save the codebook, svm and classes dicionary
        np.save(
            os.path.join("models", "bovw", model_name, f"bovw_codebook_acc{round(self.metrics['accuracy'][0]*100)}.npy"),
            self.codebook, allow_pickle=True)
        joblib.dump(self.svm, os.path.join("models", "bovw", model_name, f"bovw_svm_acc{round(self.metrics['accuracy'][0]*100)}.joblib"))
        with open(os.path.join("models", "bovw", model_name, "bovw_classes.pkl"), "wb") as dict_file:
            pickle.dump(self.class_dict, dict_file)
   
    def get_model(self, path):
        """
        Loads all files necessary for reusing a pretrained model if a path is given. Otherwise initialises the necessary objects as None type to be created when training.

        Parameters
        ----------
        path : str
            Path to the files of a pretrained model.

        Returns
        -------
        codebook : numpy.ndarray
            Contains the codes for the centroids of k-means clustering.
        svm : sklearn.svm._classes.SVC
            The Support Vector Machine to be trained.
        class_dict : dict
            Dictionary to get class names by their indices and vice versa.

        """
        codebook = None
        svm = None
        class_dict = None
        
        if path is not None:
            for obj in os.listdir(path):
                if obj.endswith(".npy"):
                    codebook = np.load(
                        os.path.join(path, obj), allow_pickle=True)
                elif obj.endswith(".joblib"):
                    svm = joblib.load(os.path.join(path, obj))
                elif obj.endswith(".pkl"):
                    with open(os.path.join(path, obj), "rb") as dict_file:
                        class_dict = pickle.load(dict_file)
            print("Pretrained model loaded.")
            
            if codebook is None or svm is None or class_dict is None:
                print(f"Model files cannot be found at path \"{path}\".",
                      "Pretrained model not loaded. Please train before inference.")
            
        return codebook, svm, class_dict
        
    def get_descriptor(self, descriptor_type):
        """
        Creates the keypoint detector and descriptor extractor object.

        Parameters
        ----------
        descriptor_type : str
            The detector type to be used. One of "orb" or "sift".

        Returns
        -------
        descriptor : cv2.Feature2D (specifically cv2.ORB or cv2.SIFT)
            The keypoint detector and descriptor extractor object.

        """
        if descriptor_type == "orb": 
            descriptor = cv2.ORB_create() 
        elif descriptor_type == "sift":
            descriptor = cv2.SIFT_create()
        else:
            print("Invalid descriptor type. Must be one of \"orb\" or",
                  "\"sift\". Will be set to \"sift\" automatically.")
            descriptor = cv2.SIFT_Create()
        
        return descriptor
        
    def get_classes(self, path):
        """
        Creates a dictionary for all class names found in the given path and their corresponding indices.

        Parameters
        ----------
        path : str
            Path to the directory containing all images sorted by class names.

        Returns
        -------
        class_dict : dict
            Dictionary to get class names by their indices and vice versa.
            
        """
        class_names = os.listdir(path)        
        class_indices = list(range(len(class_names)))
        class_dict = dict(zip(class_names, class_indices))
        class_dict.update(dict(zip(class_indices, class_names)))

        return class_dict
    
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
                                                        normalize="all")
        disp.ax_.set_title("Confusion Matrix")
        plt.savefig(filepath)
    
# MAIN ------------------------------------------------------------------------    
def main():
    load_model = True
    object_type = "bottle"
    data_path = os.path.join(
        os.getcwd(), "dataset", "augmented_dataset", object_type)
    
    if load_model:
        model_path = os.path.join(os.getcwd(), "models", "bovw", object_type)
        
        model = BagOfVisualWords(model_path)
        # pred, idx, name, img = model.predict(os.path.join(data_path, "test", "images", "contamination", "013.png"))
        # pred, idx, name, img = model.predict(os.path.join(data_path, "test", "images", "print", "011.png"))
        # print(f"Predicted \"{name}\" with {pred[0][idx]*100:.2f} % confidence.")
        model.val_dataset = model.load_all(os.path.join(data_path, "test", "images"), is_test=True)
        y_pred = model.evaluate()
        print("Accuracy:", model.metrics["accuracy"][-1])
        print("Confusion matrix:", model.metrics["confusion matrix"])       
        print("Precision:", model.metrics["precision"])
        print("Recall:", model.metrics["recall"])
        print("F1 score:", model.metrics["f1"])
        filepath = os.path.join(model_path, "confusion_matrix")
        model.plot_confusion_matrix(model.val_dataset[1], y_pred, filepath)
        # ----- For hazelnut: -------------------------------------------------
        # Accuracy: 0.7137096774193549
        # Confusion matrix: [[46  3  3  0  0]
        #                    [ 1 25  9  4  0]
        #                    [22  3 41  1  0]
        #                    [12  5  8 27  0]
        #                    [ 0  0  0  0 38]]
        # Precision: 0.7400080016778633
        # Recall: 0.7137096774193549
        # F1 score: 0.7109692388733931
        # ----- For bottle: ---------------------------------------------------
        # Accuracy: 0.6020942408376964
        # Confusion matrix: [[27 17  8  0]
        #                    [ 7 27 13  5]
        #                    [ 0  2 27 23]
        #                    [ 0  0  1 34]]
        # Precision: 0.626505059628692
        # Recall: 0.6020942408376964
        # F1 score: 0.5949857941750475
    else:
        model = BagOfVisualWords()
        model.load_data(data_path)
        model.train(svm_type="rbf", svm_iter=-1, k=200, k_iter=3)
        print("Accuracy:", model.metrics["accuracy"][-1])
        print("Confusion matrix:", model.metrics["confusion matrix"])
        model.save_model(model_name="hazelnut")

if __name__ == '__main__':
    main()