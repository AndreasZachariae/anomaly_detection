import os
import shutil
import numpy as np


def create_folders(mvtec_path,
                   augmented_path,
                   types,
                   train_percentage,
                   test_percentage,
                   val_percentage):
    if not os.path.exists(mvtec_path):
        print("ERROR: mvtec_anomaly_dataset folder not in expected path")
        return

    for type_name in os.listdir(mvtec_path):
        if type_name not in types:
            continue

        if type_name.endswith('.txt'):
            continue

        # Copy all images from mv_tec/test and /ground_truth to augmented_dataset
        for class_name in os.listdir(os.path.join(mvtec_path, type_name, "test")):

            test_class_path = os.path.join(augmented_path, type_name, "test", "images",  class_name)
            train_class_path = os.path.join(augmented_path, type_name, "train", "images", class_name)
            val_class_path = os.path.join(augmented_path, type_name, "validate", "images",  class_name)
            gt_test_class_path = os.path.join(augmented_path, type_name, "test", "ground_truth", class_name)
            gt_train_class_path = os.path.join(augmented_path, type_name, "train", "ground_truth", class_name)
            gt_val_class_path = os.path.join(augmented_path, type_name, "validate", "ground_truth", class_name)

            try:
                os.makedirs(test_class_path)
                os.makedirs(train_class_path)
                os.makedirs(val_class_path)
                os.makedirs(gt_test_class_path)
                os.makedirs(gt_train_class_path)
                os.makedirs(gt_val_class_path)

                print("Created folders for " + type_name + "/" + class_name + " in augmented dataset")
            except OSError as e:
                print(e)

            file_list = np.array(sorted(os.listdir(os.path.join(mvtec_path, type_name, "test", class_name))))
            train_files, test_files, val_files = random_split(train_percentage=0.6,
                                                              test_percentage=0.2,
                                                              val_percentage=0.2,
                                                              file_list=file_list)

            for split_folder, path, gt_path in [(train_files, train_class_path, gt_train_class_path),
                                                (test_files, test_class_path, gt_test_class_path),
                                                (val_files, val_class_path, gt_val_class_path)]:
                for file_name in split_folder:
                    src = os.path.join(mvtec_path, type_name, "test", class_name, file_name)
                    dest = os.path.join(path, file_name)
                    shutil.copy2(src, dest)

                    if class_name == "good":
                        continue

                    elements = file_name.split('.')
                    mask_name = elements[0] + "_mask" + "." + elements[1]
                    gt_src = os.path.join(mvtec_path, type_name, "ground_truth", class_name, mask_name)
                    gt_dest = os.path.join(gt_path, mask_name)
                    shutil.copy2(gt_src, gt_dest)

        # Copy all images from mv_tec/train/good to augmented_dataset
        file_list = np.array(sorted(os.listdir(os.path.join(mvtec_path, type_name, "train", "good"))))
        train_files, test_files, val_files = random_split(train_percentage,
                                                          test_percentage,
                                                          val_percentage,
                                                          file_list)

        for split_folder, split_path in [(train_files, "train"), (test_files, "test"), (val_files, "validate")]:
            for file_name in split_folder:
                elements = file_name.split('.')
                new_name = elements[0] + "_train" + "." + elements[1]

                src = os.path.join(mvtec_path, type_name, "train", "good", file_name)
                dest = os.path.join(augmented_path, type_name,  split_path, "images", "good", new_name)
                shutil.copy2(src, dest)


def random_split(train_percentage, test_percentage, val_percentage, file_list):
    p = np.random.permutation(len(file_list))
    file_list = file_list[p]

    train_split = int(len(file_list)*train_percentage)
    test_split = int(len(file_list)*(test_percentage+train_percentage))
    train_files = file_list[:train_split]
    test_files = file_list[train_split:test_split]
    val_files = file_list[test_split:]

    return train_files, test_files, val_files


def main():
    cwd = os.getcwd()
    mvtec_path = os.path.join(cwd, "dataset", "mvtec_anomaly_dataset")
    augmented_path = os.path.join(cwd, "dataset", "augmented_dataset")
    types = ["bottle", "hazelnut"]

    create_folders(mvtec_path,
                   augmented_path,
                   types,
                   train_percentage=0.6,
                   test_percentage=0.2,
                   val_percentage=0.2)


if __name__ == '__main__':
    main()
