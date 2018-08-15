import pickle


def save_to_pickle_file(python_obj, filename):
    """ Saves python object to pickle file (to avoid the need to always preprocess) """

    # TODO: add some defensive code to avoid errors with filepath
    filename = "pickles/" + filename

    with open(filename, "wb") as f:
        pickle.dump(python_obj, f)


def load_from_pickle_file(filepath):
    """ Loads a pickle file into a python variable """

    with open(filepath, "rb") as f:
        python_obj = pickle.load(f)

    return python_obj

