# -*- coding: utf-8 -*-
"""
@author: Konstantinos Flevaris
"""
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy import stats
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from sklearn.preprocessing import RobustScaler

from ibd_biom_glycoda.config import DATASETS_DIR
from ibd_biom_glycoda.data.transformations import apply_winsorization
from ibd_biom_glycoda.utils import drop_low_abundant_GPs, add_pvalue_with_significance, get_median_iqr, get_count_percentage


def load_dataset_excel(fname, data_dir=None):
    """
    Load an Excel file from the data directory.
    
    Parameters
    ----------
    fname : str
        The filename (e.g., 'dataset.xlsx')
    data_dir : str or Path, optional
        Full path to the data directory. If None, uses ibd_biom_glycoda.config.DATA_DIR.
    
    Returns
    -------
    pd.DataFrame
        The loaded Excel file as a DataFrame
    """
    base_data_dir = DATASETS_DIR if data_dir is None else Path(data_dir)
    filepath = base_data_dir / fname
    
    if not filepath.exists():
        if base_data_dir.exists():
            available_files = sorted(p.name for p in base_data_dir.glob('*.xlsx'))
        else:
            available_files = []

        if available_files:
            available_msg = (
                'Available .xlsx files in data directory: ' + ', '.join(available_files)
            )
        else:
            available_msg = 'No .xlsx files were found in the data directory.'

        raise FileNotFoundError(
            f"Dataset file not found: {fname}\n"
            f"Expected path: {filepath}\n"
            'Raw .xlsx datasets are intentionally not distributed with this repository.\n'
            'Data can be shared upon reasonable request.\n'
            'Place approved files in the repository-root datasets/ directory.\n'
            f"{available_msg}\n"
            'See README.md for data access instructions.'
        )

    df = pd.read_excel(filepath)

    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    return df


def prepare_ibd_dataset(df, binary=True, controls=['HC', 'SC'], cases=['CD', 'UC'], glycan_threshold=0.001, winsorize=False):
    """
    Prepare IBD dataset with flexible options for either binary or multiclass classification.

    Parameters:
    df (pd.DataFrame): The input dataframe containing the raw disease data.
    binary (bool): If True, create a binary dataset with specified controls and cases.
                   If False, retain the multiclass categories: HC, SC, CD, UC.
    controls (list): List of conditions to include in the control group (for binary classification).
    cases (list): List of conditions to include in the case group (for binary classification).

    Returns:
    pd.DataFrame: A processed dataframe with either binary or multiclass labels.
    """
    df_sub = df.copy()

    disease_categories = {
        'UC': ['UC'],
        'CD': ['CD'],
        'SC': ['IBS', 'IB'],
        'HC': ['HC', 'HL', 'HF', 'HS', 'Control'],
        'Other': ['DI', 'GA', 'NS', 'OT', 'OF', 'IS', 'CP', 'CO', 'IF', 'ID', 'PI', 'PC', 'IBD-U', 'IBDU']
    }
    
    for category, diseases in disease_categories.items():
        df_sub['DISEASE'] = df_sub['DISEASE'].replace(diseases, category)

    df_sub['Cohort'] = df_sub['Cohort'].replace(['Edinburgh', 'Cedars', 'Italy', 'Maastricht'], ['UK', 'US', 'IT', 'NL'])

    # Pre-specified cutoff, not data-driven.
    age_bins = [0, 40, np.inf]
    labels = ['<40', '>40']
    df_sub['Age_Group'] = pd.cut(df_sub['Age'], bins=age_bins, labels=labels, include_lowest=True)


    oxford_mapping = {
            'GP1': 'FA1',
            'GP2': 'A2',
            'GP3': 'A2B',
            'GP4': 'FA2',
            'GP5': 'M5',
            'GP6': 'FA2B',
            'GP7': 'A2G1',
            'GP8': 'FA2_6_G1',
            'GP9': 'FA2_3_G1',
            'GP10': 'FA2_6_BG1',
            'GP11': 'FA2_3_BG1',
            'GP12': 'A2G2',
            'GP13': 'A2BG2',
            'GP14': 'FA2G2',
            'GP15': 'FA2BG2',
            'GP16': 'FA2G1S1',
            'GP17': 'A2G2S1',
            'GP18': 'FA2G2S1',
            'GP19': 'FA2BG2S1',
            'GP20': '_',
            'GP21': 'A2G2S2',
            'GP22': 'A2BG2S2',
            'GP23': 'FA2G2S2',
            'GP24': 'FA2BG2S2'
        }
    
    combined_mapping = {gp: f"{gp}_{oxford}" for gp, oxford in oxford_mapping.items()}
    df_sub = df_sub.rename(columns=combined_mapping)

    df_sub_filtered = df_sub.copy()
    df_sub_filtered, allowed_gps = drop_low_abundant_GPs(df_sub_filtered, threshold=glycan_threshold) # if threshold=0, no filtering is applied

    if winsorize:
        df_sub_filtered, stats = apply_winsorization(df_sub_filtered, col_names=allowed_gps, trim_frac=0.05)

    df_sub_filtered = df_sub_filtered.set_index('Imperial_ID')

    if binary:
        df_sub_binary = df_sub_filtered.copy()

        all_categories = set(disease_categories.keys())
        if not (set(controls).issubset(all_categories) and set(cases).issubset(all_categories)):
            raise ValueError("Controls and cases must be subsets of the defined disease categories")
        
        if set(controls) & set(cases):
            raise ValueError("Controls and cases must be mutually exclusive")
        
        df_sub_binary['DISEASE'] = df_sub_binary['DISEASE'].replace(controls, 'Control')
        df_sub_binary['DISEASE'] = df_sub_binary['DISEASE'].replace(cases, 'Case')

        # Drop diagnoses outside controls/cases (e.g. 'Other') after remapping.
        df_sub_binary = df_sub_binary[df_sub_binary['DISEASE'].isin(['Control', 'Case'])]
        
        return df_sub_binary, allowed_gps
    
    else:
        df_multiclass = df_sub_filtered[df_sub_filtered['DISEASE'].isin(['HC', 'SC', 'CD', 'UC'])]

        return df_multiclass, allowed_gps

def split_loco(df, loco_test_cohort, random_state=42):
    """
    Split data for Leave-One-Cohort-Out Cross-Validation.
    
    This is the outer loop of LOCO-CV: one cohort is held out for testing,
    all others are pooled for training.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with 'Cohort' column
    loco_test_cohort : str
        Name of cohort to use as test set
    random_state : int
        Random seed for reproducibility
    
    Returns
    -------
    df_train : pd.DataFrame
        Training data (all cohorts except loco_test_cohort)
    df_test : pd.DataFrame
        Test data (only loco_test_cohort)
    """
    df_test = df[df['Cohort'] == loco_test_cohort].copy()
    df_train = df[df['Cohort'] != loco_test_cohort].copy()
    df_train = shuffle(df_train, random_state=random_state)
    
    return df_train, df_test


def extract_serological_features(df, coda_cols):
    """Extract raw serological features (glycan peaks)."""
    return df[coda_cols].copy()


def extract_demographics(df, age_ranges=None):
    """
    Extract and encode demographic features.
    
    Returns
    -------
    Z_demographics : pd.DataFrame
        Sex (0=M, 1=F) and Age (continuous)
    Z_demographics_binarized : pd.DataFrame
        Sex and Age bins (0=young, 1=old, etc.)
    age_ranges : list
        Age bin edges used
    """
    Z_demographics = df[['Sex', 'Age']].copy()
    Z_demographics['Sex'] = Z_demographics['Sex'].map({'M': 0, 'F': 1})
    
    if age_ranges is None:
        age_ranges = [0, 40, np.inf]
    
    Z_demographics_binarized = Z_demographics.copy()
    Z_demographics_binarized['Age'] = pd.cut(
        Z_demographics['Age'],
        bins=age_ranges,
        labels=list(range(len(age_ranges) - 1)),
        include_lowest=True
    ).astype(int)
    
    return Z_demographics, Z_demographics_binarized, age_ranges


def extract_disease_labels(df):
    """
    Extract disease labels and encode them.
    
    Returns
    -------
    y_disease : np.ndarray
        Encoded disease labels
        - Binary case: Control=0, Case=1
        - Multi-class: HC/SC=0, CD=1, UC=2
    """
    if df['DISEASE'].nunique() > 2:
        y_disease = df['DISEASE'].replace({
            'HC': 0,
            'SC': 1,
            'CD': 2,
            'UC': 3
        }).infer_objects(copy=False).astype(int)
    else:
        y_disease = df['DISEASE'].map({'Control': 0, 'Case': 1}).astype(int)
    
    return y_disease.to_numpy()

def extract_glycan_age(df):
    """Return GlycanAge column for downstream metadata."""
    if 'GlycanAge' not in df.columns:
        raise KeyError("Column 'GlycanAge' is missing from the dataset.")
    return df['GlycanAge'].copy()


# ``OneHotEncoder(handle_unknown='ignore')`` encodes an unseen category as an
# all-zero row. Under LOCO the encoder is fitted on the training cohorts only, so
# every held-out participant arrives that way.
UNKNOWN_COHORT_INDEX = -1
UNKNOWN_COHORT_LABEL = 'Unencoded cohort'


def cohort_index_from_onehot(V):
    """Decode a one-hot cohort matrix into integer cohort indices.

    A bare ``argmax`` returns 0 for an all-zero row and would silently label
    every held-out participant as the first training cohort. Those rows are
    returned as ``UNKNOWN_COHORT_INDEX`` instead.

    Parameters
    ----------
    V : array-like, shape (n_samples, n_cohorts)
        One-hot encoded cohort indicators.

    Returns
    -------
    ndarray
        Cohort index per sample, ``UNKNOWN_COHORT_INDEX`` where no category was
        encoded.
    """
    V = V.values if hasattr(V, 'values') else np.asarray(V)
    if V.ndim != 2:
        raise ValueError(f"Expected a 2-D one-hot cohort matrix, got shape {V.shape}.")
    idx = np.argmax(V, axis=1)
    return np.where(np.asarray(V).sum(axis=1) == 0, UNKNOWN_COHORT_INDEX, idx)


def extract_cohort_features(df, ohe_cohort=None, fit=False):
    """
    One-hot encode cohort information.
    
    Parameters
    ----------
    df : pd.DataFrame
    ohe_cohort : OneHotEncoder or None
        Pre-fitted encoder (if fit=False)
    fit : bool
        Whether to fit a new encoder
    
    Returns
    -------
    V_cohort_encoded : np.ndarray
        One-hot encoded cohort features
    cohort_locations : list
        Cohort names
    ohe_cohort : OneHotEncoder
        The encoder (fitted or passed through)
    """
    V_cohort = df[['Cohort']].copy()
    
    if fit:
        ohe_cohort = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        V_cohort_encoded = ohe_cohort.fit_transform(V_cohort)
        cohort_locations = ohe_cohort.categories_[0].tolist()
    else:
        if ohe_cohort is not None:
            V_cohort_encoded = ohe_cohort.transform(V_cohort)
            cohort_locations = ohe_cohort.categories_[0].tolist()
        else:
            # Fallback: no encoding
            V_cohort_encoded = V_cohort.to_numpy()
            cohort_locations = V_cohort['Cohort'].unique().tolist()
    
    return V_cohort_encoded, cohort_locations, ohe_cohort


def scale_demographics(Z_demographics, scaler_Z=None, fit=False):
    """
    Scale age in demographics (Sex remains binary).
    
    Parameters
    ----------
    Z_demographics : pd.DataFrame
        Demographics with 'Sex' and 'Age' columns
    scaler_Z : RobustScaler or None
    fit : bool
        Whether to fit a new scaler
    
    Returns
    -------
    Z_demographics : np.ndarray
        Scaled demographics
    scaler_Z : RobustScaler
        The scaler (fitted or passed through)
    """
    Z_demographics = Z_demographics.copy()
    
    if fit:
        scaler_Z = RobustScaler()
        Z_demographics['Age'] = scaler_Z.fit_transform(
            Z_demographics[['Age']]
        ).flatten()
    else:
        if scaler_Z is not None:
            Z_demographics['Age'] = scaler_Z.transform(
                Z_demographics[['Age']]
            ).flatten()
    
    return Z_demographics.to_numpy(), scaler_Z


def preprocess_data(df, coda_cols,
                    age_ranges=None,
                    ohe_cohort=None, 
                    scaler_Z=None,
                    fit=False):
    """
    Preprocess data. Extracts features but leaves X raw for Pipeline.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    coda_cols : list
        Column names for serological features
    age_ranges : list, optional
        Age bin edges (default: [0, 40, inf])
    ohe_cohort : OneHotEncoder, optional
        Pre-fitted cohort encoder
    scaler_Z : RobustScaler, optional
        Pre-fitted demographics scaler
    fit : bool
        Whether to fit new encoders/scalers
    
    Returns
    -------
    X_serological : pd.DataFrame
        Raw serological features (NOT scaled - for Pipeline)
    Z_demographics : np.ndarray
        Scaled demographics (Sex, Age)
    Z_demographics_binarized : np.ndarray
        Binarized demographics (Sex, Age bins)
    y_disease : np.ndarray
        Disease labels
    V_cohort_encoded : np.ndarray
        One-hot encoded cohort
    age_ranges : list
        Age bin edges
    cohort_locations : list
        Cohort names
    ohe_cohort : OneHotEncoder
        Cohort encoder
    scaler_Z : RobustScaler
        Demographics scaler
    coda_cols : list
        Serological column names (passed through)
    """
    
    X_serological = extract_serological_features(df, coda_cols)

    Z_demographics, Z_demographics_binarized, age_ranges = extract_demographics(
        df, age_ranges
    )

    y_disease = extract_disease_labels(df)

    V_cohort_encoded, cohort_locations, ohe_cohort = extract_cohort_features(
        df, ohe_cohort, fit
    )

    Z_demographics_binarized = Z_demographics_binarized.to_numpy()

    Z_demographics, scaler_Z = scale_demographics(
        Z_demographics, scaler_Z, fit
    )
    
    return (
        X_serological,
        Z_demographics,
        Z_demographics_binarized,
        y_disease,
        V_cohort_encoded,
        age_ranges,
        cohort_locations,
        ohe_cohort,
        scaler_Z,
        coda_cols
    )

def prepare_loco_data(df, coda_cols, loco_test_cohort, random_state=42, 
                      print_set_sizes=False):
    """
    Prepare data for one LOCO-CV fold.
    
    Use this function within a loop over all cohorts for full LOCO-CV.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full dataset
    coda_cols : list
        Serological feature column names
    loco_test_cohort : str
        Cohort to use as test set
    random_state : int
        Random seed
    print_set_sizes : bool
        Whether to print split sizes
    
    Returns
    -------
    train_data : tuple
        (X_train, Z_train, Z_train_bin, y_train, V_train)
    test_data : tuple
        (X_test, Z_test, Z_test_bin, y_test, V_test)
    metadata : dict
        Contains age_ranges, cohort_locations, ohe_cohort, scaler_Z, coda_cols
    """
    
    if loco_test_cohort not in df['Cohort'].unique():
        raise ValueError(f"Test cohort '{loco_test_cohort}' not found in dataset cohorts: {df['Cohort'].unique().tolist()}")
    df_train, df_test = split_loco(df, loco_test_cohort, random_state)

    if print_set_sizes:
        print(f"Test cohort: {loco_test_cohort}")
        print(f"Train set size: {len(df_train)}")
        print(f"Test set size: {len(df_test)}")
        print(f"Train cohorts: {df_train['Cohort'].unique().tolist()}")

    # Not part of train_data/test_data; carried in metadata for post-hoc analyses.
    glycan_age_train = extract_glycan_age(df_train).to_numpy()
    glycan_age_test = extract_glycan_age(df_test).to_numpy()

    # fit=True: encoders/scalers are fitted here, not on the test cohort.
    (X_train, Z_train, Z_train_bin, y_train, V_train,
     age_ranges, cohort_locations, ohe_cohort, scaler_Z, coda_cols) = preprocess_data(
        df_train, coda_cols, fit=True
    )

    # fit=False: reuses the encoders/scalers fitted on df_train above.
    X_test, Z_test, Z_test_bin, y_test, V_test, *_ = preprocess_data(
        df_test, coda_cols, ohe_cohort=ohe_cohort, scaler_Z=scaler_Z, fit=False
    )

    train_data = (X_train, Z_train, Z_train_bin, y_train, V_train)
    test_data = (X_test, Z_test, Z_test_bin, y_test, V_test)
    
    metadata = {
        'age_ranges': age_ranges,
        'cohort_locations': cohort_locations,
        'ohe_cohort': ohe_cohort,
        'scaler_Z': scaler_Z,
        'coda_cols': coda_cols,
        'loco_test_cohort': loco_test_cohort,
        'train_cohorts': df_train['Cohort'].unique().tolist(),
        'glycan_age_train': glycan_age_train,
        'glycan_age_test': glycan_age_test
    }
    
    return train_data, test_data, metadata

def generate_demographic_comparison(
    df,
    by_disease=True,
    binary=True,
    disease_order=None,
    cohort_order=None,
    alpha=0.05,
    show_pvalues=True
):
    """
    Generate a side-by-side comparison of demographic information grouped by disease or cohort,
    including p-values with integrated significance indicators based on confidence level.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe containing 'DISEASE', 'Cohort', 'Age', and 'Sex'.
    by_disease : bool, default=True
        If True, group by disease (binary or multiclass).
        If False, group by cohort (always multiclass).
    binary : bool, default=True
        Only relevant if by_disease=True.
        - If True, assumes binary classification ('Control' vs 'Case').
        - If False, assumes multiclass (more than two disease groups).
    disease_order : list, default=None
        Optional list specifying the order of disease groups in the output.
        If None, uses default order (binary: ['Control', 'Case'], multiclass: unique values).
    cohort_order : list, default=None
        Optional list specifying the order of cohort groups in the output.
        If None, uses unique values from the dataframe.
    alpha : float, default=0.05
        Significance threshold for p-value formatting.
    show_pvalues : bool, default=True
        If True, includes a p-value column with significance indicators.
        If False, excludes the p-value column from the output.

    Returns
    -------
    pd.DataFrame
        A DataFrame with side-by-side comparison of demographic information and optionally p-values
        with significance indicators.
    """
    
    groups = []

    if by_disease:
        if binary:
            disease_groups = disease_order if disease_order is not None else ['Control', 'Case']
        else:
            disease_groups = disease_order if disease_order is not None else df['DISEASE'].unique()

        for group in disease_groups:
            group_data = df[df['DISEASE'] == group]
            age_info = get_median_iqr(group_data['Age'])
            female_info = get_count_percentage(group_data['Sex'] == 'F')

            if cohort_order is not None:
                location_info = pd.Series({
                    loc: f"{(df['Cohort'] == loc).sum()} ({(df['Cohort'] == loc).sum()/len(group_data)*100:.1f}%)"
                    for loc in cohort_order if loc in group_data['Cohort'].values
                })
            else:
                location_info = group_data['Cohort'].value_counts().apply(
                    lambda x: f"{x} ({x/len(group_data)*100:.1f}%)"
                )
            
            groups.append({
                'Group': group,
                'n': len(group_data),
                'Age (median, IQR)': age_info,
                'Females (n, %)': female_info,
                **{f'Location - {loc} (n, %)': info for loc, info in location_info.items()}
            })
        
        comparison_df = pd.DataFrame(groups).set_index('Group').T

        if show_pvalues:
            if binary:
                control_age = df[df['DISEASE'] == disease_groups[0]]['Age']
                case_age = df[df['DISEASE'] == disease_groups[1]]['Age']
                normal_control = stats.shapiro(control_age).pvalue > 0.05 if len(control_age) < 5000 else True
                normal_case = stats.shapiro(case_age).pvalue > 0.05 if len(case_age) < 5000 else True
                if normal_control and normal_case:
                    age_pvalue = stats.ttest_ind(control_age, case_age, equal_var=False).pvalue
                else:
                    age_pvalue = stats.mannwhitneyu(control_age, case_age).pvalue
            else:
                age_pvalue = stats.kruskal(*[df[df['DISEASE'] == g]['Age'] for g in disease_groups]).pvalue
            
            sex_pvalue = stats.chi2_contingency(pd.crosstab(df['DISEASE'], df['Sex']))[1]
            location_pvalue = stats.chi2_contingency(pd.crosstab(df['DISEASE'], df['Cohort']))[1]

            comparison_df['p-value'] = ''
            add_pvalue_with_significance(comparison_df, 'Age (median, IQR)', age_pvalue, alpha)
            add_pvalue_with_significance(comparison_df, 'Females (n, %)', sex_pvalue, alpha)

            cohorts_to_display = cohort_order if cohort_order is not None else df['Cohort'].unique()
            for loc in cohorts_to_display:
                add_pvalue_with_significance(comparison_df, f'Location - {loc} (n, %)', location_pvalue, alpha)

    else:
        cohort_groups = cohort_order if cohort_order is not None else df['Cohort'].unique()
        
        for group in cohort_groups:
            group_data = df[df['Cohort'] == group]
            age_info = get_median_iqr(group_data['Age'])
            female_info = get_count_percentage(group_data['Sex'] == 'F')

            if disease_order is not None:
                disease_info = pd.Series({
                    dis: f"{(group_data['DISEASE'] == dis).sum()} ({(group_data['DISEASE'] == dis).sum()/len(group_data)*100:.1f}%)"
                    for dis in disease_order if dis in group_data['DISEASE'].values
                })
            else:
                disease_info = group_data['DISEASE'].value_counts().apply(
                    lambda x: f"{x} ({x/len(group_data)*100:.1f}%)"
                )
            
            groups.append({
                'Group': group,
                'n': len(group_data),
                'Age (median, IQR)': age_info,
                'Females (n, %)': female_info,
                **{f'Group - {dis} (n, %)': info for dis, info in disease_info.items()}
            })

        comparison_df = pd.DataFrame(groups).set_index('Group').T

        if show_pvalues:
            age_pvalue = stats.kruskal(*[df[df['Cohort'] == g]['Age'] for g in cohort_groups]).pvalue
            sex_pvalue = stats.chi2_contingency(pd.crosstab(df['Cohort'], df['Sex']))[1]
            disease_pvalue = stats.chi2_contingency(pd.crosstab(df['Cohort'], df['DISEASE']))[1]

            comparison_df['p-value'] = ''
            add_pvalue_with_significance(comparison_df, 'Age (median, IQR)', age_pvalue, alpha)
            add_pvalue_with_significance(comparison_df, 'Females (n, %)', sex_pvalue, alpha)

            diseases_to_display = disease_order if disease_order is not None else df['DISEASE'].unique()
            for dis in diseases_to_display:
                add_pvalue_with_significance(comparison_df, f'Group - {dis} (n, %)', disease_pvalue, alpha)

    return comparison_df