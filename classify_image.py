from sklearn.datasets import load_files
import numpy as np
from tqdm import tqdm

from keras.utils import np_utils
from keras.preprocessing import image
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import Sequential

import pickle


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# CONSTANTS
# Store list of lesion names
lesion_names = ['Melanoma', 'Nevus', 'Seborrheic Keratosis']

# METHODS
def _load_dataset(path):
    """ Function to load train / test / validation datasets """
    data = load_files(path)
    image_files = np.array(data['filenames'])
    image_targets = np_utils.to_categorical(np.array(data['target']), 3)
    return image_files, image_targets


def _path_to_tensor(img_path):
    """ Reshapes an image into the format required by keras """
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def _paths_to_tensor(img_paths):
    """ Runs the method to reshape images for keras on all images """
    list_of_tensors = [_path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


def _save_to_pickle_file(python_obj, filename):
    """ Saves python object to pickle file (to avoid the need to always preprocess) """

    # TODO: add some defensive code to avoid errors with filepath
    filename = "pickles/" + filename

    with open(filename, "wb") as f:
        pickle.dump(python_obj, f)


def _load_from_pickle_file(filepath, variable_name):
    """ Loads a pickle file into a python variable """

    with open(filepath, "rb") as f:
        variable_name = pickle.load(f)

    return variable_name


def _create_classification_model():
    """ Create a CNN in keras """
    # TODO: pass dictionary of parameters for grid search of hyperparameters

    model = Sequential()

    # First cluster of 2 x convolution and 1 x maxpool layers
    # Rationale:
    #    - increase the depth to detect different features using convolutional layers
    #    - reduce the height and width (i.e. spatial information) by using a maxpool layer
    #    - add a dropout to reduce overfitting
    model.add(Conv2D(16, (2, 2), padding='same', input_shape=train_tensors.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Second cluster of 2 x convolution and 1 x maxpool layers
    # Rationale: same as before, reduce spatial dimensions but increase depth for features
    model.add(Conv2D(64, (2, 2), padding='same', input_shape=train_tensors.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Add densely connected layers
    # Rationale: run through some fully connected layers to classify features. Dim is 133 as 133 different types of dogs
    model.add(Flatten())
    model.add(Dense(133, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(133, activation='softmax'))

    model.summary()

    return model

print("Loading image files from disk: ")
train_files, train_targets = _load_dataset("data/train")
valid_files, valid_targets = _load_dataset("data/valid")
test_files, test_targets = _load_dataset("data/test")

# print statistics about the dataset
print('There are %d total lesion categories.' % len(lesion_names))
print('There are %s total images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training images.' % len(train_files))
print('There are %d validation images.' % len(valid_files))
print('There are %d test images.'% len(test_files))

print("Reshaping images for input to keras model: ")
train_tensors = _paths_to_tensor(train_files).astype('float32')/255
valid_tensors = _paths_to_tensor(valid_files).astype('float32')/255
test_tensors = _paths_to_tensor(test_files).astype('float32')/255

# TODO: add code to export as pickle so we don't have to preprocess every time

print("Generating model architecture")
model = _create_classification_model()
