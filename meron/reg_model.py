import configparser

from keras.models import load_model
import tensorflow as tf
from tensorflow.compat.v1.keras import backend as K
from tensorflow.compat.v1 import Session


config_file = "meron_api/apps/meron_production/config/config.ini"
config_params = configparser.ConfigParser()
config_params.read(config_file)


def regress_predict(feature_vals):
    """Regression model to predict WFH or WFL

    This model takes the extracted image featuers and predics a WFH (WFL) score.

    Parameters
    ----------
    img : numpy array
          Numpy array (RGB integer format) of processed image

    Returns
    -------
    score: float
           WFH or WFL continuous zscore value

    """
    with Session(graph=tf.Graph()) as sess:

        K.set_session(sess)

        # ------------------------------------
        # Pre-trained model for classification
        # ------------------------------------
        model = load_model(config_params["files"]["regression_model"])

        # ---------------------------
        # Process image for input to
        # Keras model
        # ---------------------------
        score = model.predict(feature_vals)[0][0]

        return score
