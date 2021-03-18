import numpy as np
import configparser

from keras.models import load_model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from tensorflow.compat.v1.keras import backend as K
import tensorflow as tf
from tensorflow.compat.v1 import Session


config_file = "meron_api/apps/meron_production/config/config.ini"
config_params = configparser.ConfigParser()
config_params.read(config_file)


def extract_features(img):
    """Function to extract features from image file.

    This function extracts features from an image file using a pre-trained Resnet CNN. The features
    are then used in another model to determine WFH (WFL) score or malnutrition classification.

    Parameters
    ----------
    img : numpy array
          Numpy array (RGB integer format) of processed image


    Returns
    -------
    features : numpy array
               Numpy array (shape = (1, 2048)) of features

    """
    with Session(graph=tf.Graph()) as sess:

        K.set_session(sess)

        model = load_model(config_params["files"]["feature_model"])

        # ---------------------------
        # Process image for input to
        # Keras model
        # ---------------------------
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        features = model.predict(x, verbose=1)

        return features
