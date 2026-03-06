# config.py
RANDOM_STATE = 42

# paths
DATA_PATH = "data/sentimentdataset.csv"
MODEL_PATH = "models/sentiment_model.pkl"
OUTPUT_PATH = "outputs/"

# dataset
TEST_SIZE = 0.2

# vectorizer
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)

# cross validation
CV_FOLDS = 5

# threshold optimization
THRESHOLD_MIN = 0.1
THRESHOLD_MAX = 0.9
THRESHOLD_STEPS = 50