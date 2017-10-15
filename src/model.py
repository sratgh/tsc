from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from preprocessing import get_data
from tqdm import tqdm

# def read_images(training_folder, test_folder, image_format="png", size=(32, 32), grayscale=True, extract_labels=True):
#     '''
#     Reads in images from the given path and tries to extract labels if set to true
#     '''
#     paths_training, y = find_images(training_folder, image_format, extract_labels)
#     paths_test, y_test = find_images(test_folder, image_format, extract_labels)
#     if grayscale:
#         X = np.empty([0, size[0], size[1], 1], dtype=np.int32)
#         X_test = np.empty([0, size[0], size[1], 1], dtype=np.int32)
#     else:
#         X = np.empty([0, size[0], size[1], 3], dtype=np.int32)
#         X_test = np.empty([0, size[0], size[1], 3], dtype=np.int32)

#     for image_path in tqdm(paths_training):
#         image = read_image(image_path)
#         X = np.append(X, np.expand_dims(image, axis=0), axis=0)

#     for image_path in tqdm(paths_test):
#         image = read_image(image_path)
#         X_test = np.append(X, np.expand_dims(image, axis=0), axis=0)        

#     # One-Hot encode test vector
#     y_test = np_utils.to_categorical(y_test, 43)

#     return X, X_test, y, y_test

def split_train_validation(X, y):
    '''
    Splits the given set of samples into a training and a validation/test set
    '''
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=99)
    y_train = np_utils.to_categorical(y_train, 43)
    y_val = np_utils.to_categorical(y_val, 43)

    return X_train, X_val, y_train, y_val

def create_model():
    '''
    Creates the deep neural network
    '''
    model = Sequential()
    model.add(Conv2D(filters=8, kernel_size=2, strides=1, padding='same', activation='relu', input_shape=(32, 32, 1)))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Conv2D(filters=16, kernel_size=2, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Conv2D(filters=32, kernel_size=2, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Flatten(input_shape=(512,)))
    model.add(Dropout(0.5))
    model.add(Dense(43, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.summary()

    return model

def train_model(model, X_train, X_val, y_train, y_val, save_path='model/traffic_sign_classifier.weights.best.hdf5'):
    '''
    Trains the neural network
    '''
    checkpointer = ModelCheckpoint(filepath=save_path, verbose=1, save_best_only=True)
    model.fit(X_train, y_train, epochs=20, batch_size=30, validation_data=(X_val, y_val), callbacks=[checkpointer], verbose=1, shuffle=False)
    model.load_weights(save_path)

    return model


def predict_label(model, image):
    '''
    Predicts a label from the given image data
    '''
    return np.argmax(model.predict(X)) 

def predict_labels(model, X):
    '''
    Predicts labels from given image data
    '''
    return [predict_label(model, image) for image in tqdm(X)]


def print_results(train_accuracy, validation_accuracy, test_accuracy):
    '''
    Prints the test results 
    - accuracy
    - confusion matrix
    - train, validation and test errors
    '''
    print('Test accuracy: %.4f%%' % test_accuracy)

def get_accuracy(y_hat, y_pred):
    '''
    returns the accuracy
    '''
    print(y_hat.shape)
    return 100 * np.sum(np.array(y_pred)==np.argmax(y_hat, axis=1))/len(y_pred)

if __name__ == '__main__':

    # 1. Read in images (training and testing) and corresponding labels
    X, X_test, y, y_test = get_data(folder_training="data/GTSRB/Final_Training/Images/", folder_testing="data/GTSRB/Final_Test/Images/")

    # 2. Split into training and validation set
    X_train, X_val, y_train, y_val = split_train_validation(X, y)

    # 3. Create model and optimizer
    model = create_model()

    # 4. Train model with cross entropy and cross validation
    trained_model = train_model(model, X_train, X_val, y_train, y_val)

    # 5. Run test set on model
    y_pred = predict_labels(trained_model, X_test)
    print(get_accuracy(y_test, y_pred)) 
    # 6. Print out results (Confusion matrix, accuracy, training, validation and testing error)
    #get_accuracy()
    #get_accuracy()
    #get_accuracy()
    
    #print_results()