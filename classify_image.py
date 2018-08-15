from sklearn.datasets import load_files
import numpy as np
import pickle
from tqdm import tqdm

from keras.models import load_model
from keras.utils import np_utils
from keras.preprocessing import image
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import Sequential

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import flask

# initialize our Flask application
app = flask.Flask(__name__)


def load_cnn_model():
    """ Loads the model using the keras hdf5 file saved to disk """
    return load_model('saved_models/final_trained_model.h5')


def classify_image():
    """ Classifies an image provided by the user
     NB: this service is exposed as an endpoint """
    pass

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_cnn_model()
    app.run(host='0.0.0.0')  # add host arg so it's available on external addresses