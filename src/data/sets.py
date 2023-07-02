def show_unique_values(df, column_name):
    unique_values = df[column_name].unique()
    unique_counts = df[column_name].value_counts()

    print("Unique values in column '{}':".format(column_name))
    for value in unique_values:
        count = unique_counts[value]
        print("{}: {}".format(value, count))


# Cleaning data -> converting to float
def convert_columns_to_float(df):
    # Get a list of columns to convert
    columns_to_convert = []
    for column in df.columns:
        if df[column].dtype != 'float64':
            columns_to_convert.append(column)

    # Convert columns to float64
    df[columns_to_convert] = df[columns_to_convert].astype('float64')

    return df


# Split predictors and target variable for modeling
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


# Split into train test and val for modeling
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


# Load data sets for new experiment
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


# Saving data sets
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

# Missingnes
import pandas as pd

class MissingIncidentsChecker:
    def __init__(self, frame):
        self.frame = frame

    def check_missing_incidents(self):
        for variable in self.frame.columns:
            missing_incidents = self.frame[self.frame[variable].isnull() | self.frame[variable].eq(0) | self.frame[variable].isin(['N/A', 'nill', 'null'])][variable]
            if not missing_incidents.empty:
                print(f"Missing incidents for variable '{variable}':")
                for incident in missing_incidents:
                    print(incident)
                print() 

                import pandas as pd

# Count missingnes
import pandas as pd
import numpy as np

class MissingIncidentsCounter:
    def __init__(self, frame):
        self.frame = frame

    def count_missing_incidents(self):
        missing_counts = {}
        for variable in self.frame.columns:
            missing_count = self.frame[self.frame[variable].isnull() | self.frame[variable].eq(0) | self.frame[variable].isin(['N/A', 'nill', 'null']) | (self.frame[variable].isna() & self.frame[variable].notna())][variable].count()
            missing_counts[variable] = missing_count
        return missing_counts