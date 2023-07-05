def pop_target(df, target_col, to_numpy=False):
    """Extract target variable from dataframe and convert to nympy arrays if required

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe
    target_col : str
        Name of the target variable
    to_numpy : bool
        Flag stating to convert to numpy array or not

    Returns
    -------
    pd.DataFrame/Numpy array
        Subsetted Pandas dataframe containing all features
    pd.DataFrame/Numpy array
        Subsetted Pandas dataframe containing the target
    """

    df_copy = df.copy()
    target = df_copy.pop(target_col)
    
    if to_numpy:
        df_copy = df_copy.to_numpy()
        target = target.to_numpy()
    
    return df_copy, target


def split_sets_random(df, target_col=None, target=None, test_ratio=0.2, to_numpy=False):
    """Split sets randomly

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of the target column
    test_ratio : float
        Ratio used for the validation and testing sets (default: 0.2)

    Returns
    -------
    Numpy Array
        Features for the training set
    Numpy Array
        Target for the training set
    Numpy Array
        Features for the validation set
    Numpy Array
        Target for the validation set
    Numpy Array
        Features for the testing set
    Numpy Array
        Target for the testing set
    """
    
    from sklearn.model_selection import train_test_split
    
    if (target_col is not None) & (target_col in df.columns):
      features, target = pop_target(df=df, target_col=target_col, to_numpy=to_numpy)
    elif target is not None:
      features = df
    else:
      print("Expected either target_col or target to have a value")
      return None
    
    X_data, X_test, y_data, y_test = train_test_split(features, target, test_size=test_ratio, random_state=8)
    
    val_ratio = test_ratio / (1 - test_ratio)
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=val_ratio, random_state=8)

    return X_train, y_train, X_val, y_val, X_test, y_test

def save_sets(X_train=None, y_train=None, X_val=None, y_val=None, X_test=None, y_test=None, path='../data/processed/'):
    """Save the different sets locally

    Parameters
    ----------
    X_train: Numpy Array
        Features for the training set
    y_train: Numpy Array
        Target for the training set
    X_val: Numpy Array
        Features for the validation set
    y_val: Numpy Array
        Target for the validation set
    X_test: Numpy Array
        Features for the testing set
    y_test: Numpy Array
        Target for the testing set
    path : str
        Path to the folder where the sets will be saved (default: '../data/processed/')

    Returns
    -------
    """
    import numpy as np

    if X_train is not None:
      np.save(f'{path}X_train', X_train)
    if X_val is not None:
      np.save(f'{path}X_val',   X_val)
    if X_test is not None:
      np.save(f'{path}X_test',  X_test)
    if y_train is not None:
      np.save(f'{path}y_train', y_train)
    if y_val is not None:
      np.save(f'{path}y_val',   y_val)
    if y_test is not None:
      np.save(f'{path}y_test',  y_test)


def save_sets_v2(X_train=None, y_train=None, X_val=None, y_val=None, X_test=None, y_test=None, path='../data/processed/', suffix=''):
    """Save the different sets locally

    Parameters
    ----------
    X_train: Numpy Array
        Features for the training set
    y_train: Numpy Array
        Target for the training set
    X_val: Numpy Array
        Features for the validation set
    y_val: Numpy Array
        Target for the validation set
    X_test: Numpy Array
        Features for the testing set
    y_test: Numpy Array
        Target for the testing set
    path : str
        Path to the folder where the sets will be saved (default: '../data/processed/')
    suffix : str
        Suffix to be added to the file names (default: '')

    Returns
    -------
    """
    import numpy as np

    if X_train is not None:
      np.save(f'{path}X_train{suffix}', X_train)
    if X_val is not None:
      np.save(f'{path}X_val{suffix}',   X_val)
    if X_test is not None:
      np.save(f'{path}X_test{suffix}',  X_test)
    if y_train is not None:
      np.save(f'{path}y_train{suffix}', y_train)
    if y_val is not None:
      np.save(f'{path}y_val{suffix}',   y_val)
    if y_test is not None:
      np.save(f'{path}y_test{suffix}',  y_test)



def load_sets(path='../data/processed/', val=False):
    """Load the different locally save sets

    Parameters
    ----------
    path : str
        Path to the folder where the sets are saved (default: '../data/processed/')

    Returns
    -------
    Numpy Array
        Features for the training set
    Numpy Array
        Target for the training set
    Numpy Array
        Features for the validation set
    Numpy Array
        Target for the validation set
    Numpy Array
        Features for the testing set
    Numpy Array
        Target for the testing set
    """
    import numpy as np
    import os.path

    X_train = np.load(f'{path}X_train.npy', allow_pickle=True) if os.path.isfile(f'{path}X_train.npy') else None
    X_val   = np.load(f'{path}X_val.npy'  , allow_pickle=True) if os.path.isfile(f'{path}X_val.npy')   else None
    X_test  = np.load(f'{path}X_test.npy' , allow_pickle=True) if os.path.isfile(f'{path}X_test.npy')  else None
    y_train = np.load(f'{path}y_train.npy', allow_pickle=True) if os.path.isfile(f'{path}y_train.npy') else None
    y_val   = np.load(f'{path}y_val.npy'  , allow_pickle=True) if os.path.isfile(f'{path}y_val.npy')   else None
    y_test  = np.load(f'{path}y_test.npy' , allow_pickle=True) if os.path.isfile(f'{path}y_test.npy')  else None
    
    return X_train, y_train, X_val, y_val, X_test, y_test




def load_sets_v2(path='../data/processed/', suffix=''):
    """Load the different sets from the saved files

    Parameters
    ----------
    path : str
        Path to the folder where the sets are saved (default: '../data/processed/')
    suffix : str
        Suffix added to the file names (default: '')

    Returns
    -------
    X_train : Numpy Array or None
        Loaded features for the training set, or None if the file doesn't exist
    y_train : Numpy Array or None
        Loaded target for the training set, or None if the file doesn't exist
    X_val : Numpy Array or None
        Loaded features for the validation set, or None if the file doesn't exist
    y_val : Numpy Array or None
        Loaded target for the validation set, or None if the file doesn't exist
    X_test : Numpy Array or None
        Loaded features for the testing set, or None if the file doesn't exist
    y_test : Numpy Array or None
        Loaded target for the testing set, or None if the file doesn't exist
    """
    import numpy as np
    X_train = None
    y_train = None
    X_val = None
    y_val = None
    X_test = None
    y_test = None

    try:
        X_train = np.load(f'{path}X_train{suffix}.npy')
    except FileNotFoundError:
        pass

    try:
        y_train = np.load(f'{path}y_train{suffix}.npy')
    except FileNotFoundError:
        pass

    try:
        X_val = np.load(f'{path}X_val{suffix}.npy')
    except FileNotFoundError:
        pass

    try:
        y_val = np.load(f'{path}y_val{suffix}.npy')
    except FileNotFoundError:
        pass

    try:
        X_test = np.load(f'{path}X_test{suffix}.npy')
    except FileNotFoundError:
        pass

    try:
        y_test = np.load(f'{path}y_test{suffix}.npy')
    except FileNotFoundError:
        pass

    return X_train, y_train, X_val, y_val, X_test, y_test

# Hash Encoding

import pandas as pd
import hashlib

def hash_categorical_variable(value, num_buckets):
    # Hash the categorical value and map it to a bucket/index
    hash_value = int(hashlib.md5(value.encode('utf-8')).hexdigest(), 16)
    bucket_index = hash_value % num_buckets
    return bucket_index

def apply_hashing_trick(df, column, num_buckets):
    hashed_column = df[column].apply(lambda x: hash_categorical_variable(str(x), num_buckets))
    df['hashed_' + column] = hashed_column
    return df