import numpy as np
import tensorflow as tf
import flask
import io

from keras.models import load_model
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array

from PIL import Image


# initialize our Flask application
app = flask.Flask(__name__)

# CONSTANTS
# Store list of lesion names
lesion_names = ['Melanoma', 'Nevus', 'Seborrheic Keratosis']

# USAGE
# Start the server:
# 	python launch_prediction_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@image.jpg "http://{IP-address}:5000/predict"

# PREDICTION methods
def load_cnn_model():
    """ Loads the model using the keras hdf5 file saved to disk """
    global model
    model = load_model('saved_models/final_trained_model.h5')

    global graph  # Attempt to make keras run on threads
    graph = tf.get_default_graph()


def prepare_image(image, target):
    """ Converts the image to the required format for classification """

    # If the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image


@app.route("/predict", methods=["POST"])
def classify_image():
    """ Classifies an image provided by the user
     NB: this service is exposed as an endpoint """

    # Iinitialize the data dictionary that will be returned from the view
    data = {"success": False}

    # Ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # Preprocess the image and prepare it for classification
            image = prepare_image(image, target=(224, 224))

            # classify the input image and then initialize the list
            # of predictions to return to the client

            # Code for threadsafing (required to make code run on AWS EC2)
            global graph
            with graph.as_default():
                preds = model.predict(image)[0]

            # Return most probable
            data["prediction"] = lesion_names[np.argmax(preds)]

            # Return all predictions
            preds = preds.tolist()
            data["predictions"] = {name: preds[idx] for idx, name in enumerate(lesion_names)}

            # loop over the results and add them to the list of
            # returned predictions
            # for (imagenetID, label, prob) in results[0]:
            #     r = {"label": label, "probability": float(prob)}
            #     data["predictions"].append(r)

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_cnn_model()
    app.run(host='0.0.0.0')  # add host arg so it's available on external addresses
