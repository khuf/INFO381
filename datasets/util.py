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

def to_dataframe(data):
    def categorize(item):
        return cats[int(item)] if not pd.isna(item) else item

    df = pd.DataFrame(data['data'], columns=data['feature_names'])
    for col, cats in data['categories'].items():
        df[col] = df[col].apply(categorize).astype('category')

    return df
def fetch_employment(subset='all', data_home=None, binary_race=True, usecols=[],
                dropcols=[], numeric_only=False, dropna=True):
    """Load the CPS Employment dataset
    Binarizes 'Race' to 'White' (privileged) or 'Non-white' (unprivileged). The
    other protected attribute is 'Sex' ('Male' is privileged and 'Female' is
    unprivileged). The outcome variable is 'EmploymentStatus', 'Employed' favorable, 'Unemployed' unfavorable.
    Note:
    Args:
        subset ({'train', 'test', or 'all'}, optional): Select the dataset to
            load: 'train' for the training set, 'test' for the test set, 'all'
            for both.
        data_home (string, optional): Specify another download and cache folder
        binary_race (bool, optional): Group all non-white races together.
        usecols (single label or list-like, optional): Feature column(s) to
            keep. All others are dropped.
        dropcols (single label or list-like, optional): Feature column(s) to
            drop.
        numeric_only (bool): Drop all non-numeric feature columns.
        dropna (bool): Drop rows with NAs.
    Returns:
        namedtuple: Tuple containing X, y, and sample_weights for the Adult
        dataset accessible by index or name.
    """
    if subset not in {'train', 'test', 'all'}:
        raise ValueError("subset must be either 'train', 'test', or 'all'; "
                         "cannot be {}".format(subset))
    df = to_dataframe(fetch_openml(data_id=1590, target_column=None,
                                   data_home=data_home or DATA_HOME_DEFAULT))
    if subset == 'train':
        df = df.iloc[9879:]
    elif subset == 'test':
        df = df.iloc[:9879]

    df['EmploymentStatus'] = df['EmploymentStatus'].cat.as_ordered()# 'Employed', 'Unemployed'

    # binarize protected attributes
    if binary_race:
        df.Race = df.Race.cat.set_categories(['Non-white', 'White'],
                                             ordered=True).fillna('Non-white')
    df.Sex = df.Sex.cat.as_ordered()  # 'Female' < 'Male'

    return standardize_dataset(df, prot_attr=['Race', 'Sex'],
                              target='EmploymentStatus', sample_weight=None,
                              usecols=usecols, dropcols=dropcols,
                              numeric_only=numeric_only, dropna=dropna)


# See: https://github.com/IBM/AIF360/blob/master/aif360/algorithms/preprocessing/optim_preproc_helpers/data_preproc_functions.py
def preprocess_employment(protected_attributes=None, sub_samp=False, balance=False):
    def custom_preprocessing(df):

        df['age_by_decade'] = df['Age'].apply(lambda x: x//10*10)

        #Declare an ordering. This might need some tweaking.
        cat = pd.Categorical(df.Education, categories=["No high school diploma", "High school", "Some college, no degree",\
                "Professional degree", "Associate degree", "Bachelor's degree", "Master's degree", "Doctorate degree"], ordered=True)

        labels, unique = pd.factorize(cat, sort=True)

        # Assign labels
        df.Education = labels

        # Cut older than 70
        def age_cut(x):
            if x >= 70:
                return '>=70'
            else:
                return x

        def cut_retired_and_disabled(x):
            if x == 'Retired' or x == 'Disabled':
                return 'Ineligible'
            else:
                return x

        def group_race(x):
            if x == 'White':
                return 1.0
            else:
                return 0.0

        cat = pd.Categorical(df.Industry, categories=['Professional and business services',\
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
        
        if sub_samp and not balance:
            df = df.sample(sub_samp)
        if sub_samp and balance:
            df_0 = df[df['EmploymentStatus'] == 'Employed']
            df_1 = df[df['EmploymentStatus'] == 'Unemployed'] 
            df_0 = df_0.sample(int(sub_samp/2))
            df_1 = df_1.sample(int(sub_samp/2))
            df = pd.conact([df_1, df_1])
        print(df['Sex'].unique())
        return df

    XD_features = ['age_by_decade', 'Education', 'Industry', 'Sex', 'Race', 'Citizenship']
    D_features = ['Sex', 'Race'] if protected_attributes is None else protected_attributes
    Y_features = ['EmploymentStatus']
    X_features = list(set(XD_features)-set(D_features))
    categorical_features = ['age_by_decade', 'Citizenship', 'Industry', 'Education']

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

