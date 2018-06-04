import configparser

import cv2
import numpy as np
from imutils.face_utils import FaceAligner
from keras.applications.imagenet_utils import preprocess_input
from keras.engine import Model
from keras.layers import Flatten
from keras.models import load_model
from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from rest_framework import status
from rest_framework.exceptions import APIException
from sklearn.externals import joblib

import dlib


class NoFaceDetectedException(APIException):
    status_code = status.HTTP_400_BAD_REQUEST
    default_detail = {'image': 'No face could be detected.'}


def analyze_image(image_path,
                  score=False,
                  classification=False,
                  age=None,
                  gender='',
                  config_file='meron_api/apps/meron_production/config/config.ini'):

    '''Function to determine malnutrition classification and wfh (wfl) score from facial image
    and gender and age data.

    This function is called by the MERON API and returns malnutrition status derived from facial
    image and age and gender values. The models are pre-trained. This function just runs a
    prediction from the pretrained models.

    Parameters
    ----------
    image_path : string
                 Path and name of image file.

    score : Boolean (True or False)
            Flag whether to return the WFH (or WFL) prediction from the regression model.

    classification : Boolean (True or False)
                     Flag whether to return malnutrition classification. The categories are:
                     SAM -- Severe Acute Malnutrition
                     MAM -- Moderate Acute Malnutrition
                     NORMAL -- No detected malnutrition

    age : int
          Age in months of the subject in image

    gender : 0 or 1
             Flag to indicate the gender of the subject in image
             0 = Female
             1 = Male

    config_file : string
                  Path and name of the configuration file. Configuration file has paths and names
                  of model files as well as mapping model prediction number to malnutrition
                  classification

    Returns
    -------
    rtn_vals : dictionary
               Returns a dictionary containing either WFH (WFL) score or malnutrition classification
               or both. What is returned depdends on how the _score_ and _classification_ flags
               are set:
               If score is True a WFH (WFL) value is returned
               If classification is True a malnutrition classification is returned (SAM, MAM, NORMAL)
    '''

    # -----------
    # Gender flag
    # -----------
    if 'f' in gender.lower():
        gender_flg = 0
    else:
        gender_flg = 1

    config_params = configparser.ConfigParser()
    config_params.read(config_file)

    # -----------
    # Pre-process
    # -----------
    # Detect face, normalize brightness and align face in image
    processed_img = image_preprocess(
        image_path, landmark_file=config_params['files']['landmark_file']
    )

    # ---------------
    # Create features
    # ---------------
    # Extract image features from pre-trained CNN
    img_features = extract_features(processed_img)

    img_features = np.squeeze(img_features)

    # Append gender and age as features
    img_features = np.append(img_features, np.array([gender_flg, age]))

    img_features = np.expand_dims(img_features, axis=0)

    # Apply scale/standardize to features
    feature_scaler = joblib.load(config_params['files']['scaler_model'])
    scld_features = feature_scaler.transform(img_features)

    # -------
    # Predict
    # -------
    rtn_vals = {}
    if score:
        reg_model = load_model(config_params['files']['regression_model'])
        rtn_vals['score'] = reg_model.predict(scld_features)[0][0]

    if classification:
        class_model = load_model(config_params['files']['classification_model'])
        mal_class = np.argmax(class_model.predict(scld_features))
        rtn_vals['classification'] = config_params['classification'][str(mal_class)]

    return rtn_vals


def image_preprocess(img_file, landmark_file='./data/shape_predictor_68_face_landmarks.dat'):
    '''Function to preprocess the original image.

    This function performs a variety of pre-processing steps to the input image. These steps
    include:
    1) Facial detection
    2) Facial alignment and scaling based on eye coordinates
    3) Adjustment of image contrast using CLAHE

    Parameters
    ----------

    img_file : string
               Path and name of image file to be pre-processed

    landmark_file : string
                    Path and name of model file to facial pose estimator (see reference)


    Returns
    -------
    aligned_img : numpy array
                  Numpy array (RGB integer format) of processed image


    Reference
    ---------
    One Millisecond Face Alignment with an Ensemble of Regression Trees by Vahid Kazemi and
    Josephine Sullivan, CVPR 2014 and was trained on the iBUG 300-W face landmark dataset (see
    https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/): C. Sagonas, E. Antonakos,
    G, Tzimiropoulos, S. Zafeiriou, M. Pantic. 300 faces In-the-wild challenge: Database and
    results. Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark Localisation
    "In-The-Wild". 2016.

    '''

    # --------------------------------------------
    # Initialize dlib's face detector (HOG-based)
    # Create the facial landmark predictor and the
    # face aligner
    # --------------------------------------------
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(landmark_file)
    fa = FaceAligner(predictor, desiredFaceWidth=224)

    # --------------------------------------------------------------
    # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    # to normalize contrast of image
    # --------------------------------------------------------------
    # openCV reads in image as BGR, while Keras's reader reads RGB
    img = cv2.imread(img_file)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5)
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    img_norm = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Detect faces -- We only detect one face (first)
    rect = detector(img_norm, 1)

    # --------------------------------------------------
    # extract the ROI of the *original* face, then align
    # the face using facial landmarks
    # --------------------------------------------------
    try:
        aligned_img = fa.align(img_norm, img_norm, rect[0])
    except IndexError:
        raise NoFaceDetectedException

    return aligned_img


def extract_features(img):
    '''Function to extract features from image file.

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

    '''

    # ------------------------------
    # Pre-trained model for features
    # ------------------------------
    # Extract model for transfer learning
    vgg_model = VGGFace(model='resnet50')
    last_layer = vgg_model.get_layer('avg_pool').output
    out = Flatten(name='flatten')(last_layer)

    extractor = Model(vgg_model.input, out)

    # Freeze all layers, since we're using as fixed feature extractor
    for layer in extractor.layers:
        layer.trainable = False

    # Fixed features so optimizer and loss functions are irrelavent
    extractor.compile(optimizer='Adam', loss='categorical_crossentropy')

    # ---------------------------
    # Process image for input to
    # Keras model
    # ---------------------------
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features = extractor.predict(x, verbose=1)

    return features
