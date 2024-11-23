import os
import shutil
import random

def split_dataset(source_dir, train_dir, test_dir, train_ratio=0.7):
    """
    Splits the dataset of images into training and testing sets.

    Parameters:
    - source_dir (str): Path to the main directory containing subdirectories of dinosaur images.
    - train_dir (str): Path to the directory where the training set will be stored.
    - test_dir (str): Path to the directory where the testing set will be stored.
    - train_ratio (float): The proportion of data to be used for training (default is 0.7).

    Returns:
    - None
    """
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    dinosaur_classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    for dinosaur in dinosaur_classes:
        dinosaur_source = os.path.join(source_dir, dinosaur)
        dinosaur_train = os.path.join(train_dir, dinosaur)
        dinosaur_test = os.path.join(test_dir, dinosaur)

        os.makedirs(dinosaur_train, exist_ok=True)
        os.makedirs(dinosaur_test, exist_ok=True)

        image_files = [f for f in os.listdir(dinosaur_source)
                       if os.path.isfile(os.path.join(dinosaur_source, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

        random.shuffle(image_files)
        split_index = int(len(image_files) * train_ratio)

        train_files = image_files[:split_index]
        test_files = image_files[split_index:]

        for file_name in train_files:
            src_file = os.path.join(dinosaur_source, file_name)
            dst_file = os.path.join(dinosaur_train, file_name)
            shutil.copy2(src_file, dst_file)

        for file_name in test_files:
            src_file = os.path.join(dinosaur_source, file_name)
            dst_file = os.path.join(dinosaur_test, file_name)
            shutil.copy2(src_file, dst_file)

        print(f"Processed '{dinosaur}': {len(train_files)} training images, {len(test_files)} testing images.")

source_directory = 'Data/DinozaurImageData/dinosaur_dataset_all/' 
train_directory = 'Data/DinozaurImageData/train/'         
test_directory = 'Data/DinozaurImageData/test/'           

split_dataset(source_directory, train_directory, test_directory)
