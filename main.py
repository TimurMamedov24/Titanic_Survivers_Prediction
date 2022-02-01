import pandas as pd
from models.DataProcessing import DataProcessing
from models.Models import Model

if __name__ == "__main__":
    test = pd.read_csv("./data/test.csv")
    train = pd.read_csv("./data/train.csv")
    test_y = pd.read_csv("./data/gender_submission.csv")

    # Processed data object
    new_data = DataProcessing(train, test, test_y)
    # Start training/testing
    Model(new_data.X, new_data.y, new_data.X_test, new_data.Y_test, True)
