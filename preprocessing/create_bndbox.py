# -*- coding: utf-8 -*-

import os
import cv2
from xml.etree.ElementTree import Element, SubElement, ElementTree


class BoxCreator():    
    def __init__(self, path):
        self.path = path
        
        
    def annotate(self):
        """
        Finds bounding boxes for masked ground truth images and stores their upper-left and lower-right coordinates in XML files.

        Returns
        -------
        None.

        """
        for obj_type in os.listdir(self.path):
            type_path = os.path.join(self.path, obj_type)
            for file in os.listdir(type_path):
                if file.endswith('.png'):
                    img_path = os.path.join(type_path, file)
                    img = cv2.imread(img_path)
                    bboxes = self.get_boxPoints(img, obj_type)
                    self.create_xml(img_path, img, bboxes)
                    
        
    def get_boxPoints(self, img, obj_class):
        """
        Finds bounding boxes for a masked ground truth image.

        Parameters
        ----------
        img : numpy.ndarray
            The masked ground truth image.
        obj_class : str
            The object's class to be found in the image.

        Returns
        -------
        bboxes : list
            A list containing the class name and the coordinates for the upper-left and lower-right corner for each object instance.

        """
        bboxes = list()
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.threshold(img,128,255,cv2.THRESH_BINARY)[1]
            
        cntrs = cv2.findContours(img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        cntrs = cntrs[0] if len(cntrs)==2 else cntrs[1]
        
        for c in cntrs:
            x,y,w,h = cv2.boundingRect(c)
            bboxes.append([obj_class, x, y, x+w, y+h])
            
        return bboxes
    
    
    def create_xml(self, path, img, bboxes):
        """
        Creates the XML annotation file.

        Parameters
        ----------
        path : str
            Path to the masked ground truth images.
        img : nupmy.ndarray
            Specific masked ground truth image to get the data from.
        bboxes : list
            A list containing the class name and the coordinates for the upper-left and lower-right corner for each object instance.

        Returns
        -------
        None.

        """
        path_split = path.split(os.sep)
        proj = "anomaly_detection\\"
        proj_split = path.split(proj, 1) 
        proj_path = os.path.join(proj, proj_split[-1])
        
        annotation = Element("annotation")
        
        folder = SubElement(annotation, "folder")
        folder.text = path_split[-2]
        
        filename = SubElement(annotation, "filename")
        filename.text = path_split[-1].replace("_mask", "")
        
        path_elem = SubElement(annotation, "path")
        path_elem.text = proj_path.replace("ground_truth", "images").replace("_mask", "")
        
        source = SubElement(annotation, "source")
        database = SubElement(source, "database")
        database.text = "Unspecified"
        
        size = SubElement(annotation, "size")
        (height, width, depth) = img.shape
        w = SubElement(size, "width")
        w.text = str(width)
        h = SubElement(size, "height")
        h.text = str(height)
        d = SubElement(size, "depth")
        d.text = str(depth)
        
        for b in bboxes:
            obj = SubElement(annotation, "object")
            
            name = SubElement(obj, "name")
            name.text = str(b[0])
            pose = SubElement(obj, "pose")
            pose.text = "Unspecified"
            trunc = SubElement(obj, "truncated")
            trunc.text = "Unspecified"
            diff = SubElement(obj, "difficult")
            diff.text = "Unspecified"
            
            bbox = SubElement(obj, "bndbox")
            xmin = SubElement(bbox, "xmin")
            xmin.text = str(b[1])
            ymin = SubElement(bbox, "ymin")
            ymin.text = str(b[2])
            xmax = SubElement(bbox, "xmax")
            xmax.text = str(b[3])
            ymax = SubElement(bbox, "ymax")
            ymax.text = str(b[4])
        
        tree = ElementTree(annotation)
        xml_path = path.replace(".png", "_annotation.xml")
        tree.write(xml_path)
        
        

def main():
    gt_path = os.path.join(os.getcwd(), "dataset", "augmented_dataset", "bottle", "ground_truth")
    
    creator = BoxCreator(gt_path)
    creator.annotate()
            

if __name__ == '__main__':
    main()