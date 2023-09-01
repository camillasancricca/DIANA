import logging
from sys import exit
import pandas as pd

logging.getLogger(__name__)

__author__ = "Rayid Ghani, Pedro Saleiro <saleiro@uchicago.edu>"
__copyright__ = "Copyright \xa9 2018. The University of Chicago. All Rights Reserved."


def get_attr_cols(df):
    """
    :param df: A data frame of model results
    :param non_attr_cols: Names of columns not associated with attributes
    :return: List of columns associated with sample attributes
    """
    # index of the columns that are associated with attributes
    attr_cols = df.columns
    if attr_cols.empty:
        raise ValueError
    return attr_cols.tolist()

def preprocess_input_df(df, required_cols=None):
    """

    :param df: A data frame of model results
    :param required_cols: Names of columns required for bias calculations.
        Default is None.
    :return: A data frame, list of columns associated with sample attributes
    """

    #non_attr_cols = required_cols + ['model_id', 'as_of_date', 'entity_id', 'rank_abs', 'rank_pct', 'id', 'label_value']
    #non_string_cols = df.columns[(df.dtypes != object) & (df.dtypes != str) & (~df.columns.isin(non_attr_cols))]
    #df = discretize(df, non_string_cols)
    try:
        attr_cols_input = get_attr_cols(df)
    except ValueError:
        logging.error('preprocessing.preprocess_input_df: input dataframe does not have any other columns besides required '
                      'columns. Please add attribute columns to the input df.')
    return df, attr_cols_input
