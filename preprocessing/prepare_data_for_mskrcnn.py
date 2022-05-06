import os

def main():
    path = os.path.join(os.getcwd(), "dataset", "augmented_dataset", "bottle", "ground_truth")
    for class_name in os.listdir(path):
        class_path = os.path.join(path, class_name)
        for file in os.listdir(class_path):
            if file.endswith('.png'):
                image_name = file.replace("_mask", "")
                image_name = image_name.replace(".png", "_mask.png")
                print(image_name)
                os.rename(os.path.join(class_path, file), os.path.join(class_path, image_name)) 
            
            if file.endswith('.xml'):
                mask_name = file.replace("_mask", "")
                mask_name = mask_name.replace("annotation", "mask")
                print(mask_name)
                os.rename(os.path.join(class_path, file), os.path.join(class_path, mask_name))      

if __name__ == '__main__':
    main()