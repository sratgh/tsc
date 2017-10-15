import os
import numpy as np
from glob import glob
from tqdm import tqdm
from skimage.exposure import equalize_hist, equalize_adapthist
from skimage.io import imread, imsave
from skimage.transform import resize, rotate, warp, ProjectiveTransform
#import cv2 as cv
import matplotlib.pyplot as plt
import random
import csv

def find_images(image_folder="example_images/", image_format="ppm", set_type='training'):
    '''
    Finds images on harddrive and extracts labels from folder structure or csv file.
    '''
    y=None

    if set_type == 'training':
        image_paths = glob(os.path.join(image_folder, "*/*."+image_format))
        np.random.shuffle(image_paths) # Shuffle images
        y=[]
        for path in image_paths:
            try:
                label = int(os.path.split(path)[0][-5:])
            except:
                raise ValueError("Not possible to extract class labels from folder name. Provide y label vector.")
            y.append(label)
        y=np.asarray(y)

    elif set_type == 'testing':
        image_paths=[]  
        csv_file = open(os.path.join(image_folder, "GT-final_test.csv"))
        gtReader = csv.reader(csv_file, delimiter=';')
        gtReader.next() # skip header
        for row in gtReader:
            image_paths.append(os.path.join(image_folder,row[0])) # the 0th column is the filename
            y.append(row[7]) # the 7th column is the label
        gtFile.close()
    else:
        raise ValueError("Provide valid set_type. Either training or testing")
   
    return image_paths, y

def read_image(image_path):
    '''
    Reads in image and returns it
    '''
    img = imread(image_path)
    if len(img.shape) == 2: # Grayscale without an extra channel
        img = np.expand_dims(img, axis=2)
    return img

def preprocess_images(image_paths, resize_format=(32, 32)):
    '''
    This function takes care of the following preprocessing steps
    1. Read in images with three channel RGB
    2. Resize images to given pixel format
    3. Convert images to YUV colorspace and only keep Y channel
    4. Scale numerical image values of Y channel to [0,1] range
    5. Apply global and local histogram equalization on all images
    Returns the images as a numpy array 
    '''
    # 1. Read and resize images
    X = np.empty([0, resize_format[0], resize_format[1], 3], dtype=np.int32)
    for image_path in tqdm(image_paths):
        image = resize(read_image(image_path), resize_format, mode='edge') # Pads with the edge values of array.
        X = np.append(X, np.expand_dims(image, axis=0), axis=0)

    # Put text ?!
    #cv.putText(img, 'This one!', (230, 50), font, 0.8, (0, 255, 0), 2, cv.LINE_AA)
    
    # 2. Convert RGB to YUV and only keep Y channel, which is a conversion to grayscale 
    # (https://en.wikipedia.org/wiki/YUV)
    X = 0.299 * X[:, :, :, 0] + 0.587 * X[:, :, :, 1] + 0.114 * X[:, :, :, 2]

    # 3. Scale features to a range from 0 to 1 and convert to float32
    X = (X / 255.).astype(np.float32)
      
    # 4. Apply histogram equalization 
    # 4.1 Global (https://en.wikipedia.org/wiki/Histogram_equalization)
    X = equalize_hist(X)
    # 4.2 Local (https://en.wikipedia.org/wiki/Adaptive_histogram_equalization)
    for i in tqdm(range(X.shape[0])):
       X[i] = equalize_adapthist(X[i])

    # Add one grayscale channel
    X = X.reshape(X.shape + (1,))

    return X

def augment_images(X, y):
    '''
    Augment images through
    1. Flipping
    2. Rotating
    Returns augmented images and labels
    '''
    X, y = flip_images(X, y)
    X = rotate_images(X, 30.0)
    return X, y

def flip_images(X, y):
    '''
    Flips images horizontally and/or vertically  
    Adapts class label in case class changes when image is flipped
    '''
    # Signs that can be flipped horizontally, without changing their class
    flip_horizontally = np.array([11, 12, 13, 15, 17, 18, 22, 26, 30, 35])
    # Signs that can be flipped vertically, without changing their class
    flip_vertically = np.array([1, 5, 12, 15, 17])
    # Signs that can be flipped horizontally and vertically at the same time, without changing their class
    flip_both = np.array([32, 40])
    # Signs that can be flipped horizontally while changing their class
    flip_horizontally_change = np.array([[19, 20], [33, 34], [36, 37], [38, 39], [20, 19], [34, 33], [37, 36], [39, 38]])
    # There are no signs that can be flipped vertically while changing their class
    flip_vertically_change = None

    # Create empty arrays for augmented images and labels
    X_augmented = np.empty([0, X.shape[1], X.shape[2], X.shape[3]], dtype=X.dtype)
    y_augmented = np.empty([0], dtype=y.dtype)
    
    # Iterate through all labels
    for label in tqdm(range(43)):
        # Append all images to augmented X vector
        X_augmented = np.append(X_augmented, X[y==label], axis=0)
        # Append all images, that can be flipped horizontally without changing class
        if label in flip_horizontally:
            X_augmented = np.append(X_augmented, X[y==label][:, :, ::-1, :], axis=0)
        # Append all images, that can be flipped horizontally while changing class to the current class
        if label in flip_horizontally_change[:, 0]:
            flip_class = flip_horizontally_change[flip_horizontally_change[:, 0] == label][0][1]
            X_augmented = np.append(X_augmented, X[y==flip_class][:, :, ::-1, :], axis=0)
        # Fill extended label vector with current label
        y_augmented = np.append(y_augmented, np.full((X_augmented.shape[0] - y_augmented.shape[0]), label, dtype=np.int32))
        # Append all images, that can be flipped vertically without changing class 
        # (include the images that have been flipped horizontally)
        if label in flip_vertically:
            X_augmented = np.append(X_augmented, X_augmented[y_augmented==label][:, ::-1, :, :], axis=0)
        # Fill extended label vector with current label
        y_augmented = np.append(y_augmented, np.full((X_augmented.shape[0] - y_augmented.shape[0]), label, dtype=np.int32))
        # Append all images, that can be flipped horizontally and vertically at the same time, without changing class 
        # (include images that have been flipped vertically and horizontally in seperate steps)
        if label in flip_both:
            X_augmented = np.append(X_augmented, X_augmented[y_augmented==label][:, ::-1, ::-1, :], axis=0)
        # Fill extended label vector with current label
        y_augmented = np.append(y_augmented, np.full((X_augmented.shape[0] - y_augmented.shape[0]), label, dtype=np.int32))
    
    return X_augmented, y_augmented


def rotate_images(X, max_angle=30.0):
    '''
    Rotates given images with random angle in between [-max_angle, +max_angle] in degrees
    and appends rotated images to X and y vectors
    '''

    for i in tqdm(range(X.shape[0])):
        X[i] = rotate(X[i], random.uniform(-max_angle, max_angle), mode='edge') # Pads with the edge values of array

    return X

def save_images(X, image_folder, y=None, image_paths=None, image_format="png"):
    '''
    Saves images in given folder with given names
    '''

    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    if not image_paths:
        image_paths = ["/"+"{0:05}".format(num)+".xxx" for num in range(X.shape[0])]
    #image_names = [os.path.split(image_path)[1][:-4] for image_path in image_paths]
    for idx, path in tqdm(enumerate(image_paths)):
        if y is not None: 
            label = "{0:05}".format(y[idx])
            class_folder = os.path.join(image_folder, label)
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)
        else:
            class_folder = image_folder
        imsave(os.path.join(class_folder, os.path.split(path)[1][:-3]+image_format), np.squeeze(X[idx,:,:,:], axis=2))

def get_data(folder_training="data/GTSRB/Final_Training/Images/", folder_testing="data/GTSRB/Final_Test/Images/"):
    '''
    Does all the preprocessing for training and testing set in one function
    '''
    image_paths, y = find_images(folder_training, set_type='training')
    X = preprocess_images(image_paths, resize_format=(32,32))
    X, y = augment_images(X, y)

    image_paths, y_test = find_images(folder_testing, set_type='testing')
    X_test = preprocess_images(image_paths, resize_format=(32,32))

    # One-Hot encode test vector
    y_test = np_utils.to_categorical(y_test, 43)

    return X, X_test, y, y_test

if __name__ == '__main__':

    #y = np.arange(0,43)
    image_paths, y = find_images("data/test/", extract_labels=True)
    X = preprocess_images(image_paths, resize_format=(32,32))
    X, y = augment_images(X, y)
    #plt.imshow(np.squeeze(X[0,:,:,:], axis=2), cmap='gray')
    #plt.show()
    save_images(X, "data/preprocessed_test/", y=y)


