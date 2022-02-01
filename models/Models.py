import time

import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier


class Model:

    def __init__(self, X, Y, test, test_Y, create_csv=False):
        """
        Init for training and prediction
        :param X: processed training data
        :param Y: processed training label
        :param test: test data
        :param test_Y: test label
        :param create_csv: csv creation identificator
        """
        self.X = X
        self.Y = Y
        self.test = test
        self.test_Y = test_Y
        self.accuracy_dict = {}

        # Fetching best parameters for SVC
        cs_linear = self.tuning_linear_SVC(X, Y, test, test_Y)
        gamma_poly = self.tuning_non_linear_SVC(X, Y, test, test_Y)

        self.models = {
            "SVC_Linear_Best": SVC(kernel='linear', C=cs_linear),
            "SVC_Poly_Best": SVC(kernel='poly', gamma=gamma_poly),
            "KNeighborsN=200_P=1": KNeighborsClassifier(n_neighbors=200, p=1),
            "KNeighborsN=100_P=2": KNeighborsClassifier(n_neighbors=100, p=2),
            "KNeighborsN=50_P=3": KNeighborsClassifier(n_neighbors=50, p=3),
            "KNeighborsN=30_P=4": KNeighborsClassifier(n_neighbors=30, p=4),
            "KNeighborsP=1": KNeighborsClassifier(p=1),
            "KNeighborsP=2": KNeighborsClassifier(p=2),
            "KNeighborsP=3": KNeighborsClassifier(p=3),
            "KNeighborsP=4": KNeighborsClassifier(p=4),
            "GB150": GradientBoostingClassifier(n_estimators=150, learning_rate=0.01, max_depth=1),
            "GB100": GradientBoostingClassifier(n_estimators=100, learning_rate=0.01, max_depth=1),
            "GB50": GradientBoostingClassifier(n_estimators=50, learning_rate=0.05, max_depth=1),
            "GB10": GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, max_depth=1)
        }
        self.generate_predictions(create_csv)
        print(self.accuracy_dict)

    def generate_predictions(self, create_csv):
        """
        Method to generate predictions for models in self.models
        :param create_csv: csv creation identificator
        """
        for key, value in self.models.items():
            print(f"{key} starting")
            start = time.time()
            value.fit(self.X, self.Y)
            stop = time.time() - start
            print(f"{key} stopping, time taken:{stop}")

            predict = value.predict(self.test)
            accuracy = accuracy_score(self.test_Y['Survived'], predict)
            print(f'Accuracy of {key} = {accuracy}')
            data = {'PassengerId': self.test_Y['PassengerId'], 'Survived': predict}
            df = pd.DataFrame(data=data)
            filename = "./results/outcome" + key + ".csv"
            if create_csv:
                df.to_csv(filename, index=False)
            self.accuracy_dict[key] = accuracy

    def tuning_linear_SVC(self, X, Y, test, test_Y):
        """
        Method to tune linear SVC
        :param X: X dataset
        :param Y: Y labels
        :param test: test
        :param test_Y: test labels
        """
        cs = [0.1, 1, 10, 100]
        highest_accuracy = 0
        highest_accuracy_parameter_c = 0
        for c in cs:
            print(f'Starting Linear SVC with C={c}')
            name = "Linear.C=" + str(c)
            svc = SVC(kernel='linear', C=c)
            svc.fit(X, Y)
            print(f'Finished training, predicting...')
            prediction = svc.predict(test)
            accuracy = accuracy_score(test_Y['Survived'], prediction)
            self.accuracy_dict[name] = accuracy
            print(f'Finished predicting accuracy={accuracy}')
            if accuracy > highest_accuracy:
                highest_accuracy = accuracy
                highest_accuracy_parameter_c = c
        return highest_accuracy_parameter_c

    def tuning_non_linear_SVC(self, X, Y, test, test_Y):
        """
            Method to tune linear SVC
            :param X: X dataset
            :param Y: Y labels
            :param test: test
            :param test_Y: test labels
        """
        gammas = [0.01, 0.1, 0.5, 1]
        highest_accuracy_poly = 0
        highest_accuracy_parameter_poly = 0
        for gamma in gammas:
            print(f'Starting Polynomial SVC with Gamma={gamma}')
            name = "Poly.Gamma=" + str(gamma)
            svc_poly = SVC(kernel='poly', gamma=gamma)
            svc_poly.fit(X, Y)
            print(f'Finished training, predicting...')
            prediction_poly = svc_poly.predict(test)
            accuracy = accuracy_score(test_Y['Survived'], prediction_poly)
            self.accuracy_dict[name] = accuracy
            print(f'Finished predicting accuracy={accuracy}')
            if accuracy > highest_accuracy_poly:
                highest_accuracy_poly = accuracy
                highest_accuracy_parameter_poly = gamma
        return highest_accuracy_parameter_poly
