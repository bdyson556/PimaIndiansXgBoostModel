import sys

from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging

# Configure logger
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# log lower levels to stdout
stdout_handler = logging.StreamHandler(stream=sys.stdout)
stdout_handler.addFilter(lambda rec: rec.levelno <= logging.INFO)
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)
# log higher levels to stderr (red)
stderr_handler = logging.StreamHandler(stream=sys.stderr)
stderr_handler.addFilter(lambda rec: rec.levelno > logging.INFO)
stderr_handler.setFormatter(formatter)
logger.addHandler(stderr_handler)


def main():

    # Load and split data
    xs, ys = load_data_to_NumPy()
    x_train, x_test, y_train, y_test = split_data(xs, ys)
    logger.info("Successfully loaded and split data.")

    # Build model
    logger.info("Building the model...")
    model = train_model(x_train, y_train)
    logger.info(f"Successfully built the model. Model:\n\t{model}")

    # Make predictions
    predictions = get_binary_predictions(model, x_test)

    # Evaluate model
    accuracy_score = evaluate_model(predictions, y_test)
    logger.info("===== ACCURACY SCORE: %.2f%% =====" % (accuracy_score * 100.0))

    print(predictions)
    print(y_test)

    # for i in range(len(predictions)):
    #     print(f"\tPrediction\tActual value")
    #     comparison = f"\t{predictions[i]}\t{y_test[i]}"
    #     print(comparison)


def load_data_to_NumPy():
    # load data as NumpyArray
    dataset = loadtxt("data/pima_diabetes_raw_data.csv", delimiter=",")
    # split data into X and y (features and labels)
    xs = dataset[:, 0:8]  # Could maybe parameterize these indices.
    ys = dataset[:, 8]
    return xs, ys


def split_data(xs, ys):
    # split data into train and test sets
    seed = 7
    test_size = 0.33
    return train_test_split(xs, ys, test_size=test_size, random_state=seed)


def train_model(x_train, y_train):
    # fit model no training data
    model = XGBClassifier()  # XGBoost provides a wrapper class to allow models to be treated like classifiers or regressors in the scikit-learn framework.
    model.fit(x_train, y_train)
    return model


def get_binary_predictions(model, x_test):
    y_pred = model.predict(x_test)
    return [round(value) for value in y_pred]  # Be default, predictions are continuous values. Rounding converts to binary classification result.


def evaluate_model(predictions, y_test):
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    return accuracy


if __name__ == "__main__":
    main()