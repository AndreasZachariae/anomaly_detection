import os
import numpy as np
import pandas as pd

class DataLoader():
    def __init__(self, path):
        self.path = path
        

    def get_list_of_files(self):
        """
        Create list of all images
        Needed for labeling
        :return: csv file
        """
        path = self.path
        df = pd.DataFrame(columns=['File', 'Video', 'Path'])  # create emtpy DataFrame
        for root, dir, files in os.walk(path):
            files = [f for f in files]
            paths = [os.path.join(root, f) for f in files]
            folders = [os.path.basename(os.path.dirname(p)) for p in paths]

            df1 = pd.DataFrame({'File': files,
                                'Video': folders,
                                'Path': paths
                                })

            df = df.append(df1)
        # df = df[df['Video'].str.contains('_images')]  # Filter only files in Folder with _Images
        # df['File'] = df['File'].str.rstrip('.jpeg')  # Remove data type ending
        # df['Video'] = df['Video'].str[:-7]  # Remove '_images' from folder name
        # df['Frame_ID'] = df['File'].str.split('-', 1).str[1]  # Get Back the frame number from file name
        df['Label'] = ''
        return df


def main():
    path = r'..\dataset\mvtec_anomaly_detection'

    loader = DataLoader(path)
    df = loader.get_list_of_files()

    df.to_csv(r'..\dataset\dataset.csv')


if __name__ == "__main__":
    main()
