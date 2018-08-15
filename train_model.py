from sklearn.datasets import load_files
import numpy as np
import pickle
from tqdm import tqdm

from keras.utils import np_utils
from keras.preprocessing import image
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator # for image augmentation

from utils import load_from_pickle_file

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# TRAIN methods
def _create_classification_model(train_tensors):
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

    # Print a summary of the model
    model.summary()

    return model


def _compile_model(model):

    """ Compile the CNN """
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def _augment_input_data(train_tensors):
    """ Augments the image data to improve model accuracy """

    # create and configure augmented image generator
    datagen_train = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        rotation_range=45,
        horizontal_flip=True)

    # fit augmented image generator on data
    datagen_train.fit(train_tensors)


def _train_model_on_data(
        model,
        train_tensors,
        train_targets,
        valid_tensors,
        valid_targets,
        epochs=5,
        batch_size=20):
    """ Train the neural network """

    # Create a checkpointer to save the best weights to file
    checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5',
                                   verbose=1, save_best_only=True)

    print("Training the model: ")
    model.fit(train_tensors, train_targets,
              validation_data=(valid_tensors, valid_targets),
              epochs=epochs, batch_size=batch_size, callbacks=[checkpointer], verbose=1)


def train_model(epochs=5, batch_size=20):
    """ Trains the model using Keras and saves best fit to disk """

    print("Loading the pickled training data from disk")
    train_tensors = load_from_pickle_file("pickles/train_tensors_pickle")
    train_targets = load_from_pickle_file("pickles/train_targets_pickle")
    valid_tensors = load_from_pickle_file("pickles/valid_tensors_pickle")
    valid_targets = load_from_pickle_file("pickles/valid_targets_pickle")

    print("Neural network architecture: ")
    # TODO: expand functionality here to allow dynamic config / use of transfer learning
    model = _create_classification_model(train_tensors)

    # Compile the model
    model = _compile_model(model)

    # Augment the image data
    _augment_input_data(train_tensors)

    # Train the model
    _train_model_on_data(model, train_tensors, train_targets, valid_tensors, valid_targets, epochs, batch_size)

    print("Training complete. Model weights saved at saved_models/weights.best.from_scratch.hdf5")


def test_model():
    """ Tests the model using the test set of images """
    pass

if __name__ == "__main__":

    print("Training and testing the model")
    train_model()
    test_model()

