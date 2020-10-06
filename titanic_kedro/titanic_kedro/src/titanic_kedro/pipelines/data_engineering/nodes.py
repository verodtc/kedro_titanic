import pandas as pd
import re
import numpy as np

def merge_train_test(
        train_raw: pd.DataFrame, test_raw: pd.DataFrame
        ) -> pd.DataFrame:
    ''' Combines train_raw and test_raw datasets.

    Args:
        train_raw:  raw dataset with train data
        test_raw: raw dataset with test data

    Returns:
        raw_train_test
    '''
    # Columns to lowercase
    train_raw.columns = map(str.lower, train_raw.columns)
    test_raw.columns = map(str.lower, test_raw.columns)

    #Merging training and test
    test_raw.insert(1, 'survived', -99)
    raw_train_test = pd.concat([train_raw, test_raw])
    return raw_train_test


def clean_cabin(
        train_test_raw: pd.DataFrame
        ) -> pd.DataFrame:

    ''' Cleans cabin to keep only the corresponding letter.

    Args:
        train_test_raw

    Returns:
        train_test_intermediate
    '''
    X = train_test_raw
    X.cabin = X.cabin.fillna('U')
    X['cabin_type'] = X.cabin.map(lambda x:
                                re.compile("([a-zA-Z])").search(x).group()
                                )
    train_test_intermediate = X.drop(['cabin'], axis=1)
    return train_test_intermediate


def get_titles(
        train_test_intermediate: pd.DataFrame
        ) -> pd.DataFrame:

    ''' Get titles using name column

    Args:
        train_test_intermediate

    Returns:
        train_test_clean
    '''

    X = train_test_intermediate

    # Getting titles
    X['titles'] = X.name.str.extract(' ([A-Za-z]+)\.', expand = False)

    # Title category
    X['title_category'] = np.NaN
    X['title_category'] = np.where (X.titles.isin(['Countess',
                                                    'Jonkheer',
                                                    'Sir',
                                                   'Dame',
                                                   'L',
                                                   'Lady',
                                                   'Don',
                                                   'Dona']),
                                                   'Noble', X.title_category)

    X['title_category'] = np.where (X.titles.isin(['Major',
                                                   'Col',
                                                   'Capt']),
                                                   'Military', X.title_category)

    X['title_category'] = np.where (X.titles.isin(['Rev']),
                                                   'Religion', X.title_category)

    X['title_category'] = np.where (X.titles.isin(['Dr']),
                                                   'Academic', X.title_category)

    X['title_category'] = np.where (X.titles.isin(['Miss','Mlle']),
                                                   'Single', X.title_category)

    X['title_category'] = np.where (X.titles.isin(['Mrs','Mme']),
                                                   'Married', X.title_category)

    X['title_category'] = np.where (X.titles.isin(['Master']),
                                                   'Child', X.title_category)

    train_test_clean = X

    return train_test_clean


