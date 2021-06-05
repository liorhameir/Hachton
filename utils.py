import numpy as np
import pandas as pd
from typing import List, Union


def add_dummies_to_end(dataset: pd.DataFrame, target: str, dummy_values: Union[List, np.ndarray], true_value=1.,
                       false_value=0.) -> pd.DataFrame:
    """
    The function takes the target category and switches it to the dummy values as categories according to the
    true/false values.
    Note: the function doesn't delete the target column. in machine learning tasks it is advised to do so.

    :param dataset: pandas dataFrame
    :param target: the panda category to be switched to binary categories
    :param dummy_values: list/np.ndarray of dummy values
    :param true_value: positive dummy value
    :param false_value: negative dummy value

    :return: the new dataset (as pandas dataFrame)

    Examples:
    >>> df = pd.DataFrame({"car_color": ["blue", "green"]})
    >>> df
      car_color
    0      blue
    1     green
    >>> df = add_dummies_to_end(df, "car_color", ["blue", "green", "yellow"], 1., 0.)
    >>> df
      car_color  blue  green  yellow
    0      blue   1.0    0.0     0.0
    1     green   0.0    1.0     0.0

    """
    for dummy in dummy_values:
        dataset[str(dummy)] = dataset.apply(lambda row: true_value if row[target] == dummy else false_value, axis=1)
    return dataset


def add_dummies_to_end(dataset: pd.DataFrame, target: str, dummy_values: Union[List, np.ndarray], true_value=1.,
                       false_value=0.) -> pd.DataFrame:
    """
    The function takes the target category and switches it to the dummy values as categories according to the
    true/false values.
    Note: the function doesn't delete the target column. in machine learning tasks it is advised to do so.

    :param dataset: pandas dataFrame
    :param target: the panda category to be switched to binary categories
    :param dummy_values: list/np.ndarray of dummy values
    :param true_value: positive dummy value
    :param false_value: negative dummy value

    :return: the new dataset (as pandas dataFrame)

    Examples:
    >>> df = pd.DataFrame({"car_color": ["blue", "green"]})
    >>> df
      car_color
    0      blue
    1     green
    >>> df = add_dummies_to_end(df, "car_color", ["blue", "green", "yellow"], 1., 0.)
    >>> df
      car_color  blue  green  yellow
    0      blue   1.0    0.0     0.0
    1     green   0.0    1.0     0.0

    """
    for dummy in dummy_values:
        dataset[str(dummy)] = dataset.apply(lambda row: true_value if row[target] == dummy else false_value, axis=1)
    return dataset