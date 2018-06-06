"""Pre-load the models so they can be re-used in the views."""
import configparser

from keras.models import load_model

CONFIG_FILE = 'meron_api/apps/meron_production/config/config.ini'
config_params = configparser.ConfigParser()
config_params.read(CONFIG_FILE)


preloaded_reg_model = load_model(config_params['files']['regression_model'])
preloaded_class_model = load_model(config_params['files']['classification_model'])
