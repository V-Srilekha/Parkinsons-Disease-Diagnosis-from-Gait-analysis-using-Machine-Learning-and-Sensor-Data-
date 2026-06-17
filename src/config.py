import os

# BASE PROJECT DIRECTORY (parent of src)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# DATA DIRECTORY (contains gaitpdb folder)
DATA_DIR = os.path.join(BASE_DIR, "data", "gaitpdb")

# RESULTS DIRECTORY
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# OUTPUT FILES
MODEL_OUTPUT = os.path.join(RESULTS_DIR, "best_parkinsons_gait_model.pkl")
METRICS_OUTPUT = os.path.join(RESULTS_DIR, "model_metrics.csv")
PREDICTIONS_OUTPUT = os.path.join(RESULTS_DIR, "best_model_predictions.csv")

# TRAIN/TEST SETTINGS
TEST_SIZE = 0.20
RANDOM_STATE = 42
IMPUTE_STRATEGY = "most_frequent"

# LABELS
LABEL_PD = 1
LABEL_CONTROL = 0
