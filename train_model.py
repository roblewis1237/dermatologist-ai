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
def _create_classification_model(input_tensors):
    """ Create a CNN in keras """
    # TODO: pass dictionary of parameters for grid search of hyperparameters

    model = Sequential()

    # First cluster of 2 x convolution and 1 x maxpool layers
    # Rationale:
    #    - increase the depth to detect different features using convolutional layers
    #    - reduce the height and width (i.e. spatial information) by using a maxpool layer
    #    - add a dropout to reduce overfitting
    model.add(Conv2D(16, (2, 2), padding='same', input_shape=input_tensors.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Second cluster of 2 x convolution and 1 x maxpool layers
    # Rationale: same as before, reduce spatial dimensions but increase depth for features
    model.add(Conv2D(64, (2, 2), padding='same', input_shape=input_tensors.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Add densely connected layers
    # Rationale: run through some fully connected layers to classify features. Dim is 3 as 3 different types of lesions
    model.add(Flatten())
    model.add(Dense(6, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))

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


# TEST methods
def _test_model_accuracy(model, test_tensors, test_targets):
    """ Loads the best model from weights saved to disk and tests its accuracy 
        using the test data set """

    model.load_weights('saved_models/weights.best.from_scratch.hdf5')

    skin_lesion_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

    test_accuracy = 100 * np.sum(np.array(skin_lesion_predictions) == np.argmax(test_targets, axis=1)) / len(
        skin_lesion_predictions)

    print('Test accuracy: %.4f%%' % test_accuracy)


def _save_model_to_file(model):
    """ Saves all of the keras model to file (architecture, weights & optimiser """
    model.save('saved_models/final_trained_model.h5')

def train_and_test_model(epochs=5, batch_size=20):
    """ Trains the model using Keras and saves best fit to disk """

    print("Loading the pickled training and validation data from disk")
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

    print("Test accuracy of trained model: ")
    print("Loading the pickled testing data from disk")
    test_tensors = load_from_pickle_file("pickles/test_tensors_pickle")
    test_targets = load_from_pickle_file("pickles/test_targets_pickle")

    _test_model_accuracy(model, test_tensors, test_targets)

    _save_model_to_file(model)


if __name__ == "__main__":

    # TODO: add some command line arguments to allow e.g. specifying epochs
    print("Training and testing the model")
    train_and_test_model()


