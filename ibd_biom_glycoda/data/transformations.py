"""
@author: Konstantinos Flevaris
"""
import glycowork.glycan_data.stats as glwstats
import numpy as np
import pandas as pd
from scipy.stats import mstats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Cached glycomotif tables keyed by source files and data directory.
_GLYCOMPARE_EXCEL_CACHE = {}


def clear_glycompare_excel_cache():
    """Clear cached glycomotif tables."""
    _GLYCOMPARE_EXCEL_CACHE.clear()


def _load_glycompare_excel(abd_fname, annot_fname, data_dir=None):
    """Load the glycomotif abundance/annotation Excel files, cached by filename."""
    # Local import avoids circular dependency with dataset module
    from ibd_biom_glycoda.data.dataset import load_dataset_excel

    key = (str(abd_fname), str(annot_fname), str(data_dir))
    if key not in _GLYCOMPARE_EXCEL_CACHE:
        df_abd = load_dataset_excel(abd_fname, data_dir=data_dir)
        df_annot = load_dataset_excel(annot_fname, data_dir=data_dir)
        _GLYCOMPARE_EXCEL_CACHE[key] = (df_abd, df_annot)

    df_abd, df_annot = _GLYCOMPARE_EXCEL_CACHE[key]
    return df_abd.copy(), df_annot.copy()


def load_glycompare_data(orig_df, coda_cols, abd_fname, annot_fname, data_dir=None):
    """
    Load and process glycomotif data for comparison.

    Parameters
    ----------
    orig_df : pd.DataFrame
        The original dataframe to process
    coda_cols : list
        Column names to drop from orig_df
    abd_fname : str
        Filename for the abundance Excel file
    annot_fname : str
        Filename for the annotation Excel file
    data_dir : str or Path, optional
        Full path to the data directory. If None, uses ibd_biom_glycoda.config.DATA_DIR.

    Returns
    -------
    tuple of pd.DataFrame
        (df_glycompare, X_glycomotif) - Combined dataframe and glycomotif features
    """
    df_abd, df_annot = _load_glycompare_excel(abd_fname, annot_fname, data_dir=data_dir)

    df_glycompare = orig_df.copy()
    df_glycompare = df_glycompare.drop(coda_cols, axis=1)

    gm_to_iupac = dict(zip(df_annot['Name'], df_annot['IUPAC']))
    df_abd = df_abd.rename(columns=gm_to_iupac)

    X_glycomotif = df_abd.reset_index(drop=True)
    X_glycomotif.index = df_glycompare.index

    df_glycompare = pd.concat([df_glycompare, X_glycomotif], axis=1)

    return df_glycompare, X_glycomotif

def apply_winsorization(df, col_names, trim_frac=0.05, return_stats=True):
    """
    Winsorize each column in `col_names` by `trim_frac` at each tail,
    then re-close each row so the columns sum to the original total.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of compositional parts (each row sums to a constant).
    col_names : list[str]
        List of column names to winsorize.
    trim_frac : float, default=0.05
        Fraction to winsorize at each tail (e.g. 0.05 trims 5% at bottom & top).
    return_stats : bool, default=False
        If True, also return a dict mapping column→(n_clipped_low, n_clipped_high).

    Returns
    -------
    df2 : pd.DataFrame
        Winsorized + re-closed DataFrame
    clip_stats : dict (only if return_stats=True)
        { col_name: (n_clipped_low, n_clipped_high) }
    """
    df2 = df.copy()
    clip_stats = {}

    row_sums = df2[col_names].sum(axis=1)

    for col in col_names:
        if col not in df2:
            clip_stats[col] = (0, 0)
            continue

        mask = df2[col].notna()
        orig = df2.loc[mask, col].values

        # For counting only; mstats.winsorize below applies its own clip logic.
        lower_cut = np.percentile(orig, 100 * trim_frac)
        upper_cut = np.percentile(orig, 100 * (1 - trim_frac))

        wvals = mstats.winsorize(orig, limits=[trim_frac, trim_frac])
        df2.loc[mask, col] = wvals

        n_low  = int((orig < lower_cut).sum())
        n_high = int((orig > upper_cut).sum())
        clip_stats[col] = (n_low, n_high)

    new_sums = df2[col_names].sum(axis=1)
    df2[col_names] = df2[col_names].div(new_sums, axis=0).mul(row_sums, axis=0)

    if return_stats:
        return df2, clip_stats
    return df2

def residualize_coda(
    X_trans: pd.DataFrame,
    metadata_df: pd.DataFrame,
    covariates: list[str] | None = None,
    alpha: float = 1.0,
) -> pd.DataFrame:
    """
    Remove covariate effects from transformed compositional data using Ridge regression.
    
    Parameters
    ----------
    X_trans : pd.DataFrame
        Transformed compositional data matrix.
    metadata_df : pd.DataFrame
        Metadata containing covariate information.
    covariates : list, optional
        List of covariate column names to residualize against.
    alpha : float, default=1.0
        Ridge regression regularization parameter.
        
    Returns
    -------
    pd.DataFrame
        Residualized data with covariate effects removed.
    """
    selected_covariates = ["Age", "Sex"] if covariates is None else covariates
    cov = metadata_df.loc[X_trans.index, selected_covariates].copy()
    cov = pd.get_dummies(cov, drop_first=True)
    scaler = StandardScaler()
    X_cov = scaler.fit_transform(cov)

    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X_cov, X_trans.values)
    Y_hat = model.predict(X_cov)

    resid = X_trans.values - Y_hat
    return pd.DataFrame(resid, index=X_trans.index, columns=X_trans.columns)

def apply_clr_transformation(
    df: pd.DataFrame,
    group1: list[str] | pd.Index | None = None,
    group2: list[str] | pd.Index | None = None,
    gamma: float = 0.1,
    custom_scale: object = 0,
    random_state: int = 42,
    *,
    seed: int | None = None,
) -> pd.DataFrame:
    """Apply the glycowork CLR transformation reproducibly.

    ``seed`` remains as a backwards-compatible alias for ``random_state``.
    New analysis code should use the explicit ``random_state`` interface.
    """
    if seed is not None:
        if random_state != 42 and random_state != seed:
            raise ValueError("random_state and seed specify different values")
        random_state = seed

    glwstats.rng = np.random.default_rng(random_state)
    np.random.seed(random_state)  # in case they use legacy RNG

    # Transpose so that rows are glycans (features) and columns are samples.
    df_transposed = df.T

    if group1 is None:
        group1 = df.index

    transformed_transposed = glwstats.clr_transformation(df_transposed, group1, group2, gamma=gamma, custom_scale=custom_scale)

    # Transpose back to original orientation.
    transformed = transformed_transposed.T
    return transformed

'''
PROCESSORS
'''
class CLRProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, gamma=0.1, seed=42):
        self.gamma = gamma
        self.seed = seed

    def fit(self, X, y=None):
        # CLR preserves input column names.
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.feature_names_out_ = list(df.columns)
        return self

    def transform(self, X, y=None):
        # Returns a numpy array, not a DataFrame, for sklearn Pipeline compatibility.
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        group1 = df.index.tolist()

        glwstats.rng = np.random.default_rng(self.seed)
        np.random.seed(self.seed)

        transformed = apply_clr_transformation(
            df,
            group1=group1,
            group2=None,
            gamma=self.gamma,
            custom_scale=None,
            seed=self.seed
        )
        return transformed.values

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_, dtype=str)


class GlyCompareProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, orig_df, coda_cols, abd_fname, annot_fname, data_dir=None):
        self.orig_df = orig_df
        self.coda_cols = coda_cols
        self.abd_fname = abd_fname
        self.annot_fname = annot_fname
        self.data_dir = data_dir
        self.X_glycomotif_full = None

    def fit(self, X, y=None):
        # Loaded once here and reused by every transform() call.
        if self.X_glycomotif_full is None:
            _, self.X_glycomotif_full = load_glycompare_data(
                self.orig_df,
                self.coda_cols,
                self.abd_fname,
                self.annot_fname,
                data_dir=self.data_dir
            )
        return self

    def transform(self, X, y=None):
        # Relies on X_glycomotif_full's index matching X's.
        X_glycomotif_subset = self.X_glycomotif_full.loc[X.index]
        return X_glycomotif_subset

    def get_feature_names_out(self, input_features=None):
        return self.X_glycomotif_full.columns.tolist()
