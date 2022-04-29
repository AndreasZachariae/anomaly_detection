import os
import shutil


def main():
    cwd = os.getcwd()
    mvtec_path = os.path.join(cwd, "dataset", "mvtec_anomaly_dataset")
    augmented_path = os.path.join(cwd, "dataset", "augmented_dataset")

    if not os.path.exists(mvtec_path):
        print("ERROR: mvtec_anomaly_detection folder not in expected path")
        return

    for type_name in os.listdir(mvtec_path):
        # only use bottle dataset. remove this if all types should be copied
        if not type_name == "bottle":
            continue

        if type_name.endswith('.txt'):
            continue

        type_path = os.path.join(augmented_path, type_name)
        ground_truth_path = os.path.join(augmented_path, type_name, "ground_truth")
        images_path = os.path.join(augmented_path, type_name, "images")

        if not os.path.exists(type_path):
            os.makedirs(type_path)
            print("Created folder for type in augmented dataset")

        if not os.path.exists(ground_truth_path):
            print("Copy ground truth folder")
            shutil.copytree(os.path.join(mvtec_path, type_name, "ground_truth"), ground_truth_path)

        if not os.path.exists(images_path):
            print("Copy test folder")
            shutil.copytree(os.path.join(mvtec_path, type_name, "test"), images_path)

        # Rename exising good images to xxx_test
        for file in os.listdir(os.path.join(images_path, "good")):

            if "test" in file:
                continue

            elements = file.split('.')
            new_name = elements[0] + "_test" + "." + elements[1]

            os.rename(os.path.join(images_path, "good", file), os.path.join(images_path, "good", new_name))

        # Copy all train images to same folder as good test images
        for file in os.listdir(os.path.join(mvtec_path, type_name, "train", "good")):
            src = os.path.join(mvtec_path, type_name, "train", "good", file)
            dest = os.path.join(images_path, "good", file)
            shutil.copy2(src, dest)


if __name__ == '__main__':
    main()
