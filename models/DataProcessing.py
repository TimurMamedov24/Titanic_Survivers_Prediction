import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


class DataProcessing:

    def __init__(self, train, test, test_Y):
        self.train_x = train.copy()

        self.y = self.train_x['Survived'].values
        self.test = test.copy()
        # Group test and train together for feature engineering
        self.total = pd.concat([self.train_x, self.test], axis=0)

        # Uses total to conduct feature engineering and return train test datasets
        self.train, self.test = self.features_engineering()

        # Calculates survival rate features
        self.calculate_survival_rates()

        self.train.loc[:, 'Survival_Rate'] = (self.train['Ticket_Survival_Rate'] + self.train[
            'Family_Survival_Rate']) / 2
        self.train.loc[:, 'Survival_Rate_NA'] = (self.train['Ticket_Survival_Rate_NA'] + self.train[
            'Family_Survival_Rate_NA']) / 2

        self.test.loc[:, 'Survival_Rate'] = (self.test['Ticket_Survival_Rate'] + self.test['Family_Survival_Rate']) / 2
        self.test.loc[:, 'Survival_Rate_NA'] = (self.test['Ticket_Survival_Rate_NA'] + self.test[
            'Family_Survival_Rate_NA']) / 2

        # List of columns to drop
        self.drop_cols = ['Deck', 'Embarked', 'Family', 'Family_Size', 'Family_Size_Grouped', 'Survived',
                          'Name', 'Parch', 'Pclass', 'SibSp', 'Ticket', 'Title',
                          'Ticket_Survival_Rate', 'Family_Survival_Rate', 'Ticket_Survival_Rate_NA',
                          'Family_Survival_Rate_NA', 'Cabin']

        self.drop_cols_test = ['Deck', 'Embarked', 'Family', 'Family_Size', 'Family_Size_Grouped',
                               'Name', 'Parch', 'Pclass', 'SibSp', 'Ticket', 'Title',
                               'Ticket_Survival_Rate', 'Family_Survival_Rate', 'Ticket_Survival_Rate_NA',
                               'Family_Survival_Rate_NA', 'Cabin']

        # Non numerical features to encode
        non_numeric_features = ['Embarked', 'Sex', 'Deck', 'Title', 'Family_Size_Grouped', 'Age', 'Fare']

        # Encode labels for training
        for feature in non_numeric_features:
            self.train.loc[:, feature] = LabelEncoder().fit_transform(self.train[feature].astype(str))
            self.test.loc[:, feature] = LabelEncoder().fit_transform(self.test[feature].astype(str))
        self.X = StandardScaler().fit_transform(self.train.drop(columns=self.drop_cols))

        self.X_test = StandardScaler().fit_transform(self.test.drop(columns=self.drop_cols_test))
        self.Y_test = test_Y

    def calculate_survival_rates(self):
        """
        Method to calculate family surival rate for both test and training datasets
        """
        non_unique_families = [x for x in self.train['Family'].unique() if x in self.test['Family'].unique()]
        non_unique_tickets = [x for x in self.train['Ticket'].unique() if x in self.test['Ticket'].unique()]

        df_family_survival_rate = self.train.groupby('Family')[['Survived', 'Family', 'Family_Size']].median()
        df_ticket_survival_rate = self.train.groupby('Ticket')[['Survived', 'Ticket', 'Ticket_Frequency']].median()

        family_rates = {}
        ticket_rates = {}

        for i in range(len(df_family_survival_rate)):
            if df_family_survival_rate.index[i] in non_unique_families and df_family_survival_rate.iloc[i, 1] > 1:
                family_rates[df_family_survival_rate.index[i]] = df_family_survival_rate.iloc[i, 0]

        for i in range(len(df_ticket_survival_rate)):
            if df_ticket_survival_rate.index[i] in non_unique_tickets and df_ticket_survival_rate.iloc[i, 1] > 1:
                ticket_rates[df_ticket_survival_rate.index[i]] = df_ticket_survival_rate.iloc[i, 0]

        mean_survival_rate = np.mean(self.train['Survived'])

        train_family_survival_rate = []
        train_family_survival_rate_NA = []
        test_family_survival_rate = []
        test_family_survival_rate_NA = []

        for i in range(len(self.train)):
            if self.train['Family'][i] in family_rates:
                train_family_survival_rate.append(family_rates[self.train['Family'][i]])
                train_family_survival_rate_NA.append(1)
            else:
                train_family_survival_rate.append(mean_survival_rate)
                train_family_survival_rate_NA.append(0)

        for i in range(len(self.test)):
            if self.test['Family'].iloc[i] in family_rates:
                test_family_survival_rate.append(family_rates[self.test['Family'].iloc[i]])
                test_family_survival_rate_NA.append(1)
            else:
                test_family_survival_rate.append(mean_survival_rate)
                test_family_survival_rate_NA.append(0)

        self.train.loc[:, 'Family_Survival_Rate'] = train_family_survival_rate
        self.train.loc[:, 'Family_Survival_Rate_NA'] = train_family_survival_rate_NA
        self.test.loc[:, 'Family_Survival_Rate'] = test_family_survival_rate
        self.test.loc[:, 'Family_Survival_Rate_NA'] = test_family_survival_rate_NA

        train_ticket_survival_rate = []
        train_ticket_survival_rate_NA = []
        test_ticket_survival_rate = []
        test_ticket_survival_rate_NA = []

        for i in range(len(self.train)):
            if self.train['Ticket'][i] in ticket_rates:
                train_ticket_survival_rate.append(ticket_rates[self.train['Ticket'][i]])
                train_ticket_survival_rate_NA.append(1)
            else:
                train_ticket_survival_rate.append(mean_survival_rate)
                train_ticket_survival_rate_NA.append(0)

        for i in range(len(self.test)):
            if self.test['Ticket'].iloc[i] in ticket_rates:
                test_ticket_survival_rate.append(ticket_rates[self.test['Ticket'].iloc[i]])
                test_ticket_survival_rate_NA.append(1)
            else:
                test_ticket_survival_rate.append(mean_survival_rate)
                test_ticket_survival_rate_NA.append(0)

        self.train.loc[:, 'Ticket_Survival_Rate'] = train_ticket_survival_rate
        self.train.loc[:, 'Ticket_Survival_Rate_NA'] = train_ticket_survival_rate_NA
        self.test.loc[:, 'Ticket_Survival_Rate'] = test_ticket_survival_rate
        self.test.loc[:, 'Ticket_Survival_Rate_NA'] = test_ticket_survival_rate_NA

    def features_engineering(self):
        """
        Method to fill empty values, add new features
        :return test, train datasets with update features
        """

        # Create family column
        self.train_x['Family'] = self.get_surnames(self.train_x)
        self.train_x['Family_Size'] = self.train_x['SibSp'] + self.train_x['Parch'] + 1

        # Create Title columns
        self.train_x['Title'] = self.train_x['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
        self.train_x['Title'].replace(
            ['Miss', 'Mrs', 'Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'],
            'Miss/Mrs/Ms')
        self.train_x['Title'].replace(
            ['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'],
            'Dr/Military/Noble/Clergy')

        self.train_x['Is_Married'] = 0
        self.train_x['Is_Married'].loc[self.train_x['Title'] == 'Mrs'] = 1

        self.train_x['Deck'] = self.train_x['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')

        self.train_x.groupby(['Sex', 'Pclass'])['Age'].apply(
            lambda x: x.fillna(x.median())).reset_index(
            drop=True)

        self.train_x['Embarked'].fillna('S')

        # Fill empty slots in fare column
        third_class_fare = self.train_x.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]
        self.train_x['Fare'] = self.train_x['Fare'].fillna(third_class_fare)
        self.train_x['Fare'] = pd.qcut(self.train_x['Fare'], 13)
        self.train_x['Age'] = pd.qcut(self.train_x['Age'], 9)
        self.train_x['Ticket_Frequency'] = self.train_x.groupby('Ticket')['Ticket'].transform('count')

        # Create family size feature
        family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 5: 'Medium', 6: 'Medium', 7: 'Large', 8: 'Large',
                      11: 'Large'}
        self.train_x['Family_Size_Grouped'] = self.train_x['Family_Size'].map(family_map)

        """
        Do the same for test dataset
        """

        self.test['Family'] = self.get_surnames(self.test)
        self.test['Family_Size'] = self.test['SibSp'] + self.test['Parch'] + 1

        # Create Title columns
        self.test['Title'] = self.test['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
        self.test['Title'].replace(
            ['Miss', 'Mrs', 'Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'],
            'Miss/Mrs/Ms')
        self.test['Title'].replace(
            ['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'],
            'Dr/Military/Noble/Clergy')

        self.test['Is_Married'] = 0
        self.test['Is_Married'].loc[self.test['Title'] == 'Mrs'] = 1

        self.test['Deck'] = self.test['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')

        self.test.groupby(['Sex', 'Pclass'])['Age'].apply(
            lambda x: x.fillna(x.median())).reset_index(
            drop=True)

        self.test['Embarked'].fillna('S')

        # Fill empty slots in fare column
        third_class_fare = self.test.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]
        self.test['Fare'] = self.test['Fare'].fillna(third_class_fare)
        self.test['Fare'] = pd.qcut(self.test['Fare'], 13)
        self.test['Age'] = pd.qcut(self.test['Age'], 9)
        self.test['Ticket_Frequency'] = self.test.groupby('Ticket')['Ticket'].transform('count')

        # Create family size feature
        family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 5: 'Medium', 6: 'Medium', 7: 'Large', 8: 'Large',
                      11: 'Large'}
        self.test['Family_Size_Grouped'] = self.test['Family_Size'].map(family_map)
        return self.train_x, self.test

    def get_surnames(self, set):
        """
        Method to extract families using their name
        :return: families
        """
        names = set['Name']
        families = []

        for i in range(len(names)):
            if '(' in names.iloc[i]:
                name_no_bracket = names.iloc[i].split('(')[0]
            else:
                name_no_bracket = names.iloc[i]

            family_name = name_no_bracket.split(",")[0]
            families.append(family_name)
        return families
