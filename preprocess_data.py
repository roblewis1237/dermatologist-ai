from sklearn.datasets import load_files
import numpy as np
import pickle
from tqdm import tqdm

from keras.utils import np_utils
from keras.preprocessing import image

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from utils import save_to_pickle_file


# CONSTANTS
# Store list of lesion names
lesion_names = ['Melanoma', 'Nevus', 'Seborrheic Keratosis']

# PRE-PROCESSING methods
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


def preprocess_data(
        train_data="data/train",
        valid_data="data/valid",
        test_data="data/test"):
    """ Preprocess the image data so it can be loaded into Keras model """

    print("Loading image files from disk: ")
    train_files, train_targets = _load_dataset(train_data)
    valid_files, valid_targets = _load_dataset(valid_data)
    test_files, test_targets = _load_dataset(test_data)

    # Print come statistics about the data
    print('There are %d total lesion categories.' % len(lesion_names))
    print('There are %s total images.\n' % len(np.hstack([train_files, valid_files, test_files])))
    print('There are %d training images.' % len(train_files))
    print('There are %d validation images.' % len(valid_files))
    print('There are %d test images.' % len(test_files))

    print("Transforming data to tensor format for keras input: ")
    train_tensors = _paths_to_tensor(train_files).astype('float32') / 255
    valid_tensors = _paths_to_tensor(valid_files).astype('float32') / 255
    test_tensors = _paths_to_tensor(test_files).astype('float32') / 255

    print("Convert outputs to pickle and store to disk: ")
    save_to_pickle_file(train_tensors, "pickles/train_tensors_pickle")
    save_to_pickle_file(valid_tensors, "pickles/valid_tensors_pickle")
    save_to_pickle_file(test_tensors, "pickles/test_tensors_pickle")

    save_to_pickle_file(train_targets, "pickles/train_targets_pickle")
    save_to_pickle_file(valid_targets, "pickles/valid_targets_pickle")
    save_to_pickle_file(test_targets, "pickles/test_targets_pickle")

    print("Preprocessing complete. Pickle files saved to /pickles/ ")


if __name__ == "__main__":
    preprocess_data()
