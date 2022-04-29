import cv2
import numpy as np
import random
import os
import math


class DataAugmentation():
    def __init__(self, folder_path, multiplier):
        self.folder_path = folder_path
        self.multiplier = multiplier

    def augment_dataset(self):
        for type_name in os.listdir(self.folder_path):
            # only use bottle dataset. remove this if all types should be copied
            if not type_name == "bottle":
                continue

            for class_name in os.listdir(os.path.join(self.folder_path, type_name, "ground_truth")):
                for image_name in os.listdir(os.path.join(self.folder_path, type_name, "images", class_name)):
                    mask_name = image_name.split('.')[0] + "_mask." + image_name.split('.')[1]
                    mask_path = os.path.join(self.folder_path, type_name, "ground_truth", class_name)
                    image_path = os.path.join(self.folder_path, type_name, "images", class_name)

                    self.augment_image(image_path, image_name, mask_path, mask_name)

    def augment_image(self, image_path, image_name, mask_path, mask_name):
        original_image = cv2.imread(os.path.join(image_path, image_name))
        original_mask = cv2.imread(os.path.join(mask_path, mask_name))

        for flip in ["hflip", "vflip", "noflip"]:
            if flip == "hflip":
                image_flipped = self.horizontal_flip(original_image)
                mask_flipped = self.horizontal_flip(original_mask)
            if flip == "vflip":
                image_flipped = self.vertical_flip(original_image)
                mask_flipped = self.vertical_flip(original_mask)
            if flip == "noflip":
                image_flipped = original_image
                mask_flipped = original_mask

            for rotation in range(int(math.sqrt(self.multiplier/3))):
                angle = int(random.uniform(-90, 90))
                image_rotated = self.rotation(image_flipped, angle)
                mask_rotated = self.rotation(mask_flipped, angle, mask=True)

                for brightness in range(int(math.sqrt(self.multiplier/3))):
                    value = random.uniform(0.5, 1.5)
                    image = self.brightness(image_rotated, value)

                    self.save_image(image, image_path, image_name, flip, angle, value)
                    self.save_image(mask_rotated, mask_path, mask_name, flip, angle, value)

                    # self.show_three_images(original_image, image, mask_rotated)

    def show_three_images(self, image_original, image_augmented, mask_augmented):
        image_original = cv2.resize(image_original, (0, 0), None, .25, .25)
        image_augmented = cv2.resize(image_augmented, (0, 0), None, .25, .25)
        mask_augmented = cv2.resize(mask_augmented, (0, 0), None, .25, .25)
        image_stack = np.hstack((image_original, image_augmented, mask_augmented))
        cv2.imshow("image", image_stack)
        cv2.waitKey(0)

    def horizontal_flip(self, image):
        return cv2.flip(image, 1)

    def vertical_flip(self, image):
        return cv2.flip(image, 0)

    def brightness(self, image, value):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = np.array(hsv, dtype=np.float64)
        hsv[:, :, 1] = hsv[:, :, 1]*value
        hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
        hsv[:, :, 2] = hsv[:, :, 2]*value
        hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
        hsv = np.array(hsv, dtype=np.uint8)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return image

    def rotation(self, image, angle, mask=False):
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
        if mask:
            borderValue = (0, 0, 0)
        else:
            borderValue = (255, 255, 255)

        image = cv2.warpAffine(image,
                               M,
                               (w, h),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=borderValue)
        return image

    def save_image(self, image, path, name, flip, angle, value):
        cv2.imwrite(path + "/" +
                    name.split('.')[0] +
                    "_" + flip +
                    "_rotate" + str(angle) +
                    "_bright" + str(round(value, 3)) +
                    "." + name.split('.')[1],
                    image)


def main():
    print("DataAugmentation started")

    folder_path = os.path.join(os.getcwd(), "dataset", "augmented_dataset")

    # The effect of augmentation is an image multiplication by 3*x*x
    # Only values with the next step in this list have an increasing effect
    # multiplier = [3, 12, 29, 48, 75, 108, ...]

    da = DataAugmentation(folder_path, multiplier=12)
    da.augment_dataset()


if __name__ == '__main__':
    main()
