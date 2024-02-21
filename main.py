from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging

logger = logging.getLogger(__name__)

def main():

    # Load and split data
    xs, ys = load_data_to_NumPy()
    x_train, x_test, y_train, y_test = split_data(xs, ys)
    logger.info("Successfully loaded and split data.")

    # Build the model
    model = train_model(x_train, y_train)
    


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
    model = XGBClassifier()
    model.fit(x_train, y_train)
    return model


if __name__ == "__main__":
    main()