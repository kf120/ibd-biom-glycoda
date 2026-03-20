# -*- coding: utf-8 -*-
"""
@author: Konstantinos Flevaris
"""
import re
import random
import numpy as np
import pandas as pd
import glycowork.glycan_data.stats as glwstats

from scipy.spatial.distance import cdist, euclidean
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer

def seed_everything(seed):
    """
    Set random seed for reproducibility.

    Parameters:
    seed (int): Seed value for random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)

    # ensure glwstats uses the same RNG
    glwstats.rng = np.random.default_rng(seed)

def preprocess_data(df, bin_edges=None, fit_binning=False):
    """
    Preprocess the data by extracting features, encoding categorical variables,
    and binning continuous variables.
    Parameters:
    df (pd.DataFrame): Input dataframe.
    bin_edges (KBinsDiscretizer, optional): Fitted KBinsDiscretizer for age binning.
    fit_binning (bool): Whether to fit a new KBinsDiscretizer.
    Returns:
    tuple: Processed X, Z, y, V arrays and fitted bin_edges.
    """
    # Create copies to avoid SettingWithCopyWarning
    X_df = df.filter(like='GP').copy()
    Z_df = df[['Sex', 'Age']].copy()
    y_df = df['DISEASE'].map({'Control': 0, 'Case': 1})
    V_df = df[['Cohort']].copy()
    
    Z_df['Sex'] = Z_df['Sex'].map({'M': 0, 'F': 1})

    if fit_binning:
        bin_edges = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')
        Z_df.loc[:, 'Age'] = bin_edges.fit_transform(Z_df[['Age']]).astype(int).flatten()
    else:
        Z_df.loc[:, 'Age'] = bin_edges.transform(Z_df[['Age']]).astype(int).flatten()
    
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    V_df = ohe.fit_transform(V_df)
    
    # Convert to numpy arrays
    X = X_df.to_numpy()
    Z = Z_df.to_numpy()
    y = y_df.to_numpy()
    
    return X, Z, y, V_df, bin_edges
        
def drop_low_abundant_GPs(df, threshold=2.0, min_samples=0.5):
    """
    Drops columns containing the specified substring if most values are below the threshold,
    then renormalizes the remaining columns to sum to 100%.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    substring (str): Substring to identify columns for processing (default: 'GP')
    threshold (float): Threshold percentage for dropping columns (default: 2.0)
    min_samples (float): Minimum proportion of samples that should be above threshold (default: 0.5)
    
    Returns:
    pd.DataFrame: Processed DataFrame with low abundant GPs dropped and remaining GPs renormalized
    """
    # Identify coda columns based on the presence of specific patterns
    coda_patterns = ["GP", "A1", "A2", "M5"]
    gp_columns = [col for col in df.columns if any(pattern in col for pattern in coda_patterns)]

    df_processed = df.copy()
    
    # Calculate the proportion of samples above threshold for each GP column
    above_threshold = (df_processed[gp_columns] > threshold).mean()
    
    # Identify columns to keep (those with proportion above threshold greater than min_samples)
    columns_to_keep = above_threshold[above_threshold > min_samples].index.tolist()
    
    # Drop low abundant GP columns
    columns_to_drop = [col for col in gp_columns if col not in columns_to_keep]
    df_processed.drop(columns=columns_to_drop, inplace=True)
    
    # Renormalize remaining GP columns to sum to 100%
    gp_columns_kept = [col for col in columns_to_keep if col in df_processed.columns]
    row_sums = df_processed[gp_columns_kept].sum(axis=1)
    df_processed[gp_columns_kept] = df_processed[gp_columns_kept].div(row_sums, axis=0) * 100
    
    print(f"Dropped {len(columns_to_drop)} low abundant GP columns: {columns_to_drop}")
    print(f"Retained {len(gp_columns_kept)} GP columns: {gp_columns_kept}")
    
    return df_processed, gp_columns_kept

def natural_sort_key(glycan_name):
    """Sort glycans naturally: GP1, GP2, ..., GP10, GP11, etc."""
    parts = re.split(r'(\d+)', glycan_name)
    return [int(part) if part.isdigit() else part for part in parts]

def get_safe_index(data, indices):
    """Index data safely, handling both DataFrames and arrays."""
    return data.iloc[indices] if hasattr(data, 'iloc') else data[indices]

def safe_column_access(data, col_idx):
    """Return a column by index from a DataFrame or array.

    Parameters
    ----------
    data : DataFrame or ndarray or None
        Source array.
    col_idx : int
        Column index to extract.

    Returns
    -------
    Series or ndarray or None
        Selected column or ``None`` if unavailable.
    """
    if data is None:
        return None
    if hasattr(data, 'iloc'):
        return data.iloc[:, col_idx]
    else:
        return data[:, col_idx]

def add_pvalue_with_significance(comparison_df, row_name, pvalue, alpha=0.05):
    """
    Add formatted p-value with significance marker to DataFrame in-place.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        DataFrame to update.
    row_name : str
        Row label to update.
    pvalue : float
        P-value to format.
    alpha : float
        Significance threshold.
    """
    significance_indicator = ' (*)' if pvalue < alpha else ''
    if pvalue < 1e-3:
        comparison_df.loc[row_name, 'p-value'] = f"p < 0.001{significance_indicator}"
    else:
        comparison_df.loc[row_name, 'p-value'] = f"{pvalue:.2f}{significance_indicator}"

def get_median_iqr(x):
    """
    Return median and IQR as formatted string.

    Parameters
    ----------
    x : pd.Series
        Input series.
    """
    median = x.median()
    q1, q3 = x.quantile([0.25, 0.75])
    return f"{median:.1f} ({q1:.1f}-{q3:.1f})"


def get_count_percentage(x):
    """
    Return count and percentage as formatted string.

    Parameters
    ----------
    x : pd.Series
        Input series.
    """
    count = x.sum()
    percentage = (count / len(x)) * 100
    return f"{count} ({percentage:.1f}%)"