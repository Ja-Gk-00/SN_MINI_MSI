import numpy as np
import pandas as pd
import struct
from array import array
from os.path import join

#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        # Read labels
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(f'Magic number mismatch, expected 2049, got {magic}')
            labels = np.array(array("B", file.read()), dtype=np.uint8)
        
        # Read images
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(f'Magic number mismatch, expected 2051, got {magic}')
            image_data = np.array(array("B", file.read()), dtype=np.uint8)
            images = image_data.reshape(size, rows * cols)  # Flatten each image to 1D
            
        return images, labels
            
    def load_data(self):
        # Load training and test data
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        
        # Convert to DataFrames
        train_df = pd.DataFrame(x_train)
        train_df['label'] = y_train
        test_df = pd.DataFrame(x_test)
        test_df['label'] = y_test
        
        return train_df, test_df

# Example usage:
# Initialize the data loader with paths to the MNIST dataset files

input_path = r'MNIST'
train_images_path = join(input_path, 'train-images.idx3-ubyte')
train_labels_path = join(input_path, 'train-labels.idx1-ubyte')
test_images_path = join(input_path, 't10k-images.idx3-ubyte')
test_labels_path = join(input_path, 't10k-labels.idx1-ubyte')

mnist_loader = MnistDataloader(train_images_path, train_labels_path, test_images_path, test_labels_path)
train_df, test_df = mnist_loader.load_data()

# Display the first few rows of the training DataFrame
print(train_df.head())
