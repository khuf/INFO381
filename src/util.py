import os
import sys
sys.path.insert(0, os.path.abspath('../datasets'))

from EmploymentDataset import EmploymentDataset

import pandas as pd
from sklearn.datasets import fetch_openml
from aif360.sklearn.datasets.utils import standardize_dataset

# cache location
DATA_HOME_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 '..', 'data')

# Preprocessing 
def preprocess_employment(protected_attributes=None, sub_samp=False, balance=False):
    def custom_preprocessing(df):

        df['age_by_decade'] = df['Age'].apply(lambda x: 0 if x < 14 else x//10*10)

        #Declare an ordering. This might need some tweaking.
        cat = pd.Categorical(df.Education, categories=['missing', "No high school diploma", "High school", "Some college, no degree",\
                "Professional degree", "Associate degree", "Bachelor's degree", "Master's degree", "Doctorate degree"], ordered=True)

        labels, unique = pd.factorize(cat, sort=True)
        
        # Assign labels
        df.Education = labels

        cat = pd.Categorical(df.Region, categories=["South", "West", "Northeast", "Midwest"], ordered=False)

        labels, unique = pd.factorize(cat, sort=True)

        df.Region = labels

        cat = pd.Categorical(df.Married, categories=["Married", "Widowed", "Never Married", "Divorced", "Separated"], ordered=False)

        labels,unique = pd.factorize(cat, sort=True)


        # Assign labels
        df.Married = labels

        # Cut older than 70
        def age_cut(x):
            if x >= 70:
                return '>=70'
            elif x == 0:
                return '>=14'
            else:
                return x

        def cut_retired_and_disabled(x):
            if x == 'Retired' or x == 'Disabled':
                return 'Ineligible'
            elif x == "Not in Labor Force":
                return "Unemployed"
            else:
                return x

        def group_race(x):
            if x == 'White':
                return 1.0
            else:
                return 0.0
        def group_household(x):
            if x == 1:
                return 'single'
            elif x == 2: 
                return 'couple'
            else:
                return 'family'

        cat = pd.Categorical(df.Industry, categories=['missing', 'Professional and business services',\
                'Educational and health services', 'Transportation and utilities',\
                'Public administration', 'Leisure and hospitality', 'Trade',\
                'Manufacturing', 'Other services', 'Financial', 'Information',\
                'Construction', 'Mining','Agriculture, forestry, fishing, and hunting',\
                'Armed forces'], ordered=False)
        labels, unique = pd.factorize(cat, sort=True)
        df.Industry = labels
        def group_citizenship(x):
            if x == 'Citizen, Native':
                return 'Native'
            else:
                return 'Non Native'

        # Cluster age attributes
        df['age_by_decade'] = df['age_by_decade'].apply(lambda x: age_cut(x))
        df['Citizenship'] = df['Citizenship'].apply(lambda x: group_citizenship(x))
        df['Sex'] = df['Sex'].replace({'Female': 0.0, 'Male': 1.0})
        df['Race'] = df['Race'].apply(lambda x: group_race(x))
        df['EmploymentStatus'].apply(lambda x: cut_retired_and_disabled(x))
        df['PeopleInHousehold'].apply(lambda x: group_household(x))
        
        if sub_samp and not balance:
            df = df.sample(sub_samp)
        if sub_samp and balance:
            df_0 = df[df['EmploymentStatus'] == 'Employed']
            df_1 = df[df['EmploymentStatus'] == 'Unemployed'] 
            df_0 = df_0.sample(int(sub_samp/2))
            df_1 = df_1.sample(int(sub_samp/2))
            df = pd.conact([df_1, df_1])
        return df

    XD_features = ['age_by_decade', 'Education', 'Sex', 'Race', 'Region', 'Citizenship', 'Married', 'Industry']
    D_features = ['Sex', 'Race'] if protected_attributes is None else protected_attributes
    Y_features = ['EmploymentStatus']
    X_features = list(set(XD_features)-set(D_features))
    categorical_features = ['age_by_decade', 'Citizenship', 'Region', 'Education', 'PeopleInHousehold', 'Married', 'Industry']

    # privileged classes
    all_privileged_classes = {"Sex": [1.0], 'Race': [1.0]}

    # protected attribute maps
    all_protected_attribute_maps = {'Sex': {1.0: 'Male', 0.0: 'Female'},\
            'Race': {1.0: 'White', 0.0: 'Non White'}}

    return EmploymentDataset(
            label_name = Y_features[0],
            favorable_classes=['Employed'],
            protected_attribute_names=D_features,
            privileged_classes = [all_privileged_classes[x] for x in D_features],
            instance_weights_name = None,
            categorical_features = categorical_features,
            features_to_keep=X_features+Y_features+D_features,
            na_values=['NA'],
            metadata= {'label_maps': [{1.0: 'Employed', 0.0: 'Unemployed'}],
                'protected_attribute_maps': [all_protected_attribute_maps[x] for x in D_features]},
            custom_preprocessing=custom_preprocessing)

