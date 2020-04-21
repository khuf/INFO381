import os

import pandas as pd

from aif360.datasets import StandardDataset

default_mappings = {
        'label_maps': [{1.0: 'Employed', 0.0: 'Unemployed'}],
        'protected_attribute_maps': [{1.0: 'Male', 0.0: 'Female'},\
                {1.0: 'White', 0.0: 'Not White'}]
        }


#def default_preprocessing(df):
#    cat = pd.Categorical(df.Education, categories=["No high school diploma", "High school", "Some college, no degree",\
#                                                            "Professional degree", "Associate degree", "Bachelor's degree", "Master's degree", "Doctorate degree"], ordered=True)
#    labels, unique = pd.factorize(cat, sort=True)
#
#    # Assign labels
#    df.Education = labels
#    return df
class EmploymentDataset(StandardDataset):
    
    def __init__(self, label_name='EmploymentStatus',
            favorable_classes=['Employed'],
            protected_attribute_names=['Sex', 'Race'],
            privileged_classes=[['White'], ['Male']],
            instance_weights_name=None,
            categorical_features=['Education', 'Age', 'Hispanic', 'Industry', 'Citizenship', 'Married'],
            features_to_keep=[], features_to_drop=['CountryOfBirthCode', 'MetroAreaCode', 'PeopleInHousehold', 'Region'],
            na_values=['NA'], custom_preprocessing=None,
            metadata=default_mappings):

        train_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  '..', 'data', 'raw', 'employment.data')
        test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  '..', 'data', 'raw', 'employment.test')

        column_names = ['PeopleInHousehold', 'Region', 'State', 'MetroAreaCode', 'Age',\
                'Married',\
                'Sex',\
                'Education',\
                'Race',\
                'Hispanic',\
                'CountryOfBirthCode',\
                'Citizenship',\
                'EmploymentStatus',\
                'Industry']

        try:
            train = pd.read_csv(train_path, header=0, names=column_names,
                skipinitialspace=True, na_values=na_values)
            test = pd.read_csv(test_path, header=0, names=column_names,
                skipinitialspace=True, na_values=na_values)
        except IOError as err:
            print("IOError: {}".format(err))
            print("Ensure that dataset files exist in <project_root>/data/raw/")
            print("\n\t{}\n".format(os.path.abspath(os.path.join(
               os.path.abspath(__file__), '..', 'data', 'raw', 'employment'))))
            import sys
            sys.exit(1)

        df = pd.concat([test, train], ignore_index=True)

        super(EmploymentDataset, self).__init__(df=df, label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop, na_values=na_values,
            custom_preprocessing=custom_preprocessing, metadata=metadata)

  
