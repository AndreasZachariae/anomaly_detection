import cv2
import numpy as np
import random

class DataAugmentation():
    def __init__(self):

        self.folder_path = r"dataset/mvtec_anomaly_detection/bottle/test/broken_large"
        self.dest_path = r"dataset/augment_test"

    def augment_image(self, image_name):
        original_image = cv2.imread(self.folder_path + "/" + image_name)

        # image=self.horizontal_flip(original_image)
        # image=self.vertical_flip(original_image)

        for rotation in range(10): # np.linspace(start=-90, stop=90, num=10):
            for brightness in range(5): # np.linspace(start=0.5, stop=1.5, num=5):
                image, angle = self.rotation(original_image)
                image, value = self.brightness(original_image)
                self.save_image(image, image_name.split('.')[0] + "rotate" + str(angle) + "bright" + str(round(value,1)))

        # self.show_two_images(original_image, image)

    def show_two_images(self, image_original, image_augmented):
        image_original = cv2.resize(image_original, (0, 0), None, .25, .25)
        image_augmented = cv2.resize(image_augmented, (0, 0), None, .25, .25)
        vertical_stack = np.hstack((image_original, image_augmented))
        cv2.imshow("image", vertical_stack)
        cv2.waitKey(0)

    def horizontal_flip(self, image):
        return cv2.flip(image, 1)

    def vertical_flip(self, image):
        return cv2.flip(image, 0)

    def brightness(self, image, low=0.5, high=1.5):
        value = random.uniform(low, high)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = np.array(hsv, dtype = np.float64)
        hsv[:,:,1] = hsv[:,:,1]*value
        hsv[:,:,1][hsv[:,:,1]>255]  = 255
        hsv[:,:,2] = hsv[:,:,2]*value 
        hsv[:,:,2][hsv[:,:,2]>255]  = 255
        hsv = np.array(hsv, dtype = np.uint8)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # self.save_image(image, "bright" + str(value))
        return image, value

    def rotation(self, image, angle=90):
        angle = int(random.uniform(-angle, angle))
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
        image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
        # self.save_image(image, "rotate" + str(angle))
        return image, angle

    def save_image(self, image, name):
        cv2.imwrite(self.dest_path + "/" + name + ".jpg", image)

def main():
    print("DataAugmentation started")
    da = DataAugmentation()
    da.augment_image("000.png")
    

if __name__ == '__main__':
    main()