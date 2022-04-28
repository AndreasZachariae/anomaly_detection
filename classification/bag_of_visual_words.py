import os
import numpy as np
import collections

import cv2

from scipy.cluster.vq import kmeans,vq, whiten
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC 


class BagOfVisualWords():    
    def __init__(self, path, split_ratio=0.85):
        self.class_dict = self.get_classes(path)
        (imgs, labels) = self.load_data(path)
        (self.imgs_train, self.lbls_train, self.imgs_test, self.lbls_test) \
            = self.random_split(imgs, labels, split_ratio)
        self.descriptor = None
        self.codebook = None
        self.scaler = None
        self.svm = None
        
        
    def metrics(self):
        if (self.descriptor is None or self.codebook is None 
            or self.scaler is None or self.svm is None):
            print("Metrics cannot be calculated. Please check if training",
                  " has been done")
            return
        
        print("\nCalculating metrics...")
        print("Extract keypoint features...")
        (img_descr, _) = self.get_descriptor_lists(self.imgs_test)
        print("Create bag of codewords...")
        features = self.get_codewords(img_descr)
        
        print("Predict training data with Support Vector Machine...")
        predictions = self.svm.predict(features)
        
        accuracy = accuracy_score(self.lbls_test, predictions)
        precision = precision_score(self.lbls_test, predictions, average='weighted')
        recall = recall_score(self.lbls_test, predictions, average='weighted')
        f1 = f1_score(self.lbls_test, predictions, average='weighted')
        metrics = {"accuracy" : accuracy,
                   "precision": precision,
                   "recall"   : recall,
                   "f1"       : f1}
        # print(predictions)
        # print(self.lbls_test)
        print("Calculating done.")  
        
        return metrics

    
    def train(self, descriptor="orb", svm_type="rbf", svm_iter=80000, k=200, k_iter=20):
        print("\nTraining started...")
        self.k = k
        
        
        # TODO: test with different parameters
        if descriptor == "orb": 
            self.descriptor = cv2.ORB_create() 
        elif descriptor == "sift":
            self.descriptor = cv2.SIFT_create()
        else:
            print("Invalid descriptor. Must be one of \"orb\" or \"sift\".")
            return
        
        try:
            self.svm = SVC(kernel=svm_type, max_iter=svm_iter)
        except ValueError:
            print("Invalid parameter for the Support Vector Machine. svm_type"
                  + " must be one of \"linear\", \"poly\", \"rbf\", \"sigmoid"
                  + ", or \"precomputed\". svm_iter must be a positive integer"
                  + " or -1 for no limit.")
            return
        
        print("Extract keypoint features...")
        (img_descr, descrs) = self.get_descriptor_lists(self.imgs_train)
        print("Create codebook...")
        # rescale to give features unit variance before clustering
        descrs = whiten(descrs)
        self.codebook, _ = kmeans(descrs, k, k_iter) # TODO test with other parameters
        print("Create bag of codewords...")
        features = self.get_codewords(img_descr)
        
        print("Train Support Vector Machine...")
        self.svm.fit(features, np.array(self.lbls_train))
        
        print("Training done.")
        
    
    def get_codewords(self, img_descr_list):
        """
        Creates the bag of visual words.

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
        
        # normalise features as many machine learning estimators (e.g. SVM with 
        # rbf kernel) assume standard normally distributed data
        self.scaler = StandardScaler().fit(features) 
        features = self.scaler.transform(features)
        
        return features
        
        
    def get_descriptor_lists(self, img_paths):
        """
        Calculates feature descriptors for every image and stores them in two different ways.

        Parameters
        ----------
        img_paths : numpy.ndarray
            An array containing all image paths.

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
            _, dscr = self.descriptor.detectAndCompute(img, None)
            img_des.append([path, dscr])
            descriptors += list(dscr)
           
        img_des = np.array(img_des, dtype=object)
        descriptors = np.array(descriptors, dtype=float)
           
        return (img_des, descriptors)
    
    
    def random_split(self, images, labels, split_ratio):
        """
        Randomly splits the dataset into training and test data.

        Parameters
        ----------
        images : numpy.ndarray
            An array containing all image paths.
        labels : numpy.ndarray
            An array containing the class labels corresponding to the images.
        split_ratio : float
            The ratio of training data to training +  test data.

        Returns
        -------
        imgs_train : numpy.ndarray
            An array containing all training image paths.
        lbls_train : numpy.ndarray
            An array containing all training class labels.
        imgs_test : numpy.ndarray
            An array containing all test image paths.
        lbls_test : numpy.ndarray
            An array containing all test class labels.

        """
        # shuffle image paths and their class labels in the same way
        p = np.random.permutation(len(images))
        images = images[p]
        labels = labels[p]
        
        # split into training and test data
        split_val = int(len(images)*split_ratio)
        imgs_test = images[split_val:]
        lbls_test = labels[split_val:]
        imgs_train = images[:split_val]
        lbls_train = labels[:split_val]
        
        return (imgs_train, lbls_train, imgs_test, lbls_test)
    
        
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
        
    
    def load_data(self, path):
        """
        Make a list of all images' paths and their classes respectively.

        Parameters
        ----------
        path : str
            Path to the directory containing all images sorted by class names.

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
        num_defect_classes = float(len(occ.keys())-1)
        num_defect_per_class = 0
        for key in occ:
            if key != self.class_dict["good"]:
                num_defect_per_class += occ[key] 
            print(f"- {occ[key]} images of class \"{self.class_dict[key]}\".")
        
        # make sure there is about the same amount of good images compared to
        # other (anomalous) classes in the dataset
        mean_num = round(num_defect_per_class/num_defect_classes)
        pos_good = np.where(img_classes == self.class_dict["good"])[0]
        random_good = np.random.choice(pos_good, pos_good.shape[0]-mean_num, replace=False)
        img_paths = np.delete(img_paths, random_good)
        img_classes = np.delete(img_classes, random_good)
        print(f"Using only {mean_num} random \"good\" images.")
            
        return (img_paths, img_classes)
       

def main():
    data_path = os.path.join(os.getcwd(), "dataset", "augmented_dataset", "bottle", "images")
    
    bovw = BagOfVisualWords(data_path, split_ratio=0.6)
    bovw.train(descriptor="sift", svm_iter=-1, k=200, k_iter=3)
    metr = bovw.metrics()
    print(metr)
    # print(f"Accuracy: {metr['accuracy']}.")
            

if __name__ == '__main__':
    main()
    
        