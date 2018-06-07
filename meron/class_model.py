import configparser
import numpy as np

from keras.models import load_model
from keras import backend as K
import tensorflow as tf

config_file = 'meron_api/apps/meron_production/config/config.ini'
config_params = configparser.ConfigParser()
config_params.read(config_file)


def class_predict(feature_vals):
    '''Classification model to predict WFH or WFL

    This model takes the extracted image featuers and predicts a malnutrition classification.

    Parameters
    ----------
    img : numpy array
          Numpy array (RGB integer format) of processed image

    Returns
    -------
    score: int
           Malnutrition classification (SAM, MAM, NORMAL). The mapping of malnutrition classification
           to integer is given in the configuration file


    '''
    with tf.Session(graph=tf.Graph()) as sess:

        K.set_session(sess)

        # ---------------------------
        # Process image for input to
        # Keras model
        # ---------------------------
        model = load_model(config_params['files']['classification_model'])
        mal_class = np.argmax(model.predict(feature_vals))

        return mal_class
