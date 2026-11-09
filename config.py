from pathlib import Path
import numpy as np

#settings for training data 
DATA_PATH = Path("data")
VALID_ENDING = "validation_set"
LABEL_ENDING = "label"

#path to output
MODEL_PATH = Path("models")

#settings for predicting
INPUTS_PATH = Path("test_data")
PREDICTION_PATH = Path("predictions")
LEARNER_PATH = Path("models//lane_model.pkl")

#codes for segmentation
CODES = np.array(["background", "left_lb", "right_lb"], dtype=str)

#settings for training
BATCH_SIZE = 4
WORKERS = 0
EPOCHS = 1

#settings for loss
CE_WEIGHT = 0.95
TVERSKY_A = 0.7
TVERSKY_B = 0.3

#debug mode
DEBUG = False
SEED = 42
TRAIN_IMAGES = 90
VALID_IMAGES = 10