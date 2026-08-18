# -*- coding: utf-8 -*-
"""
@author: Konstantinos Flevaris
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from PIL import Image


def create_rgb_palette(palette_name='okabe_ito'):
    """
    Create colorblind-friendly color palettes suitable for scientific figures.
    
    All palettes are designed to be distinguishable for people with color vision
    deficiencies.
    
    Parameters
    ----------
    palette_name : str
        Name of palette: 'okabe_ito', 'tol_bright', 'tol_muted', 'tol_light',
        'seaborn_colorblind',
        'grayscale'
    
    Returns
    -------
    colors_rgb : list of tuples
        List of (R, G, B) tuples in range [0, 1] for use with matplotlib
    
    References
    ----------
    - Okabe & Ito (2008): https://jfly.uni-koeln.de/color/
    - Paul Tol: https://personal.sron.nl/~pault/
    
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> colors = create_rgb_palette('okabe_ito')
    >>> plt.scatter(x, y, c=categories, cmap=colors)
    """
    
    palettes_rgb = {
        'okabe_ito': [
            # Okabe & Ito (2008) - Most widely used colorblind-safe palette
            (0.00, 0.45, 0.70),  # Blue
            (0.90, 0.60, 0.00),  # Orange
            (0.00, 0.60, 0.50),  # Bluish green
            (0.95, 0.90, 0.25),  # Yellow
            (0.35, 0.70, 0.90),  # Sky blue
            (0.80, 0.40, 0.00),  # Vermillion
            (0.80, 0.60, 0.70),  # Reddish purple
            (0.00, 0.00, 0.00),  # Black
        ],
        
        'tol_bright': [
            # Paul Tol's Bright qualitative scheme (max 7 colors)
            (0.27, 0.51, 0.71),  # Blue
            (0.89, 0.10, 0.11),  # Red
            (0.30, 0.69, 0.29),  # Green
            (0.60, 0.31, 0.64),  # Purple
            (1.00, 0.50, 0.00),  # Orange
            (1.00, 1.00, 0.20),  # Yellow
            (0.65, 0.34, 0.16),  # Brown
        ],
        
        'tol_muted': [
            # Paul Tol's Muted qualitative scheme (max 9 colors)
            (0.20, 0.13, 0.53),  # Indigo
            (0.33, 0.66, 0.41),  # Cyan
            (0.27, 0.67, 0.60),  # Teal
            (0.07, 0.47, 0.20),  # Green
            (0.60, 0.60, 0.20),  # Olive
            (0.87, 0.80, 0.47),  # Sand
            (0.80, 0.40, 0.47),  # Rose
            (0.53, 0.13, 0.33),  # Wine
            (0.67, 0.27, 0.60),  # Purple
        ],
        
        'tol_light': [
            # Paul Tol's Light qualitative scheme (max 9 colors)
            (0.47, 0.71, 0.84),  # Light blue
            (0.99, 0.80, 0.67),  # Light orange
            (0.60, 0.89, 0.74),  # Light green
            (0.99, 0.93, 0.60),  # Light yellow
            (0.87, 0.69, 0.82),  # Light purple
            (0.99, 0.77, 0.76),  # Light pink
            (0.74, 0.74, 0.74),  # Light gray
            (0.80, 0.92, 0.77),  # Pale green
            (0.96, 0.96, 0.82),  # Pale yellow
        ],
        
        'seaborn_colorblind': [
            # Seaborn's default colorblind palette
            (0.00, 0.45, 0.70),  # Blue
            (0.90, 0.62, 0.00),  # Orange
            (0.00, 0.62, 0.45),  # Green
            (0.84, 0.37, 0.00),  # Red
            (0.34, 0.71, 0.91),  # Sky blue
            (0.80, 0.47, 0.65),  # Pink
        ],
        
        'grayscale': [
            # Grayscale palette for accessibility
            (0.00, 0.00, 0.00),  # Black
            (0.25, 0.25, 0.25),  # Dark gray
            (0.50, 0.50, 0.50),  # Medium gray
            (0.75, 0.75, 0.75),  # Light gray
            (0.90, 0.90, 0.90),  # Very light gray
            (1.00, 1.00, 1.00),  # White
        ],
    }
    
    if palette_name not in palettes_rgb:
        available = ', '.join(sorted(palettes_rgb.keys()))
        raise ValueError(f"Unknown palette: '{palette_name}'. Choose from: {available}")
    
    return palettes_rgb[palette_name]


def create_rgb_colormap(colors_rgb, name='custom_rgb', n_bins=256):
    """
    Create a matplotlib colormap from RGB color values.
    
    Parameters
    ----------
    colors_rgb : list of tuples
        List of (R, G, B) tuples in range [0, 1]
    name : str
        Name for the colormap
    n_bins : int
        Number of discrete colors in the colormap (default: 256 for smooth)
    
    Returns
    -------
    cmap : matplotlib.colors.LinearSegmentedColormap
        Colormap ready to use with matplotlib plotting functions
    
    Examples
    --------
    >>> colors_rgb = [(0.0, 0.45, 0.70), (0.9, 0.6, 0.0)]
    >>> cmap = create_rgb_colormap(colors_rgb, name='blue_orange')
    >>> plt.imshow(data, cmap=cmap)
    """
    cmap = LinearSegmentedColormap.from_list(name, colors_rgb, N=n_bins)
    return cmap


def create_cohort_colors(cohort_names, palette='okabe_ito'):
    """
    Generate consistent color mapping for cohorts.
    
    Parameters
    ----------
    cohort_names : list
        List of cohort names (e.g., ['IT', 'NL', 'UK', 'US'])
    palette : str
        RGB palette to use: 'okabe_ito', 'tol_bright', 'tol_muted', etc.
    
    Returns
    -------
    cohort_colors : dict
        Mapping of cohort names to RGB tuples
    
    Examples
    --------
    >>> cohorts = ['IT', 'NL', 'UK', 'US']
    >>> colors = create_cohort_colors(cohorts)
    >>> # {'IT': (0.0, 0.45, 0.7), 'NL': (0.9, 0.6, 0.0), ...}
    """
    color_list = create_rgb_palette(palette)
    
    # Sort cohort names to ensure consistent ordering
    sorted_cohorts = sorted(cohort_names)
    
    # Create mapping
    cohort_colors = {}
    for i, cohort in enumerate(sorted_cohorts):
        cohort_colors[cohort] = color_list[i % len(color_list)]
    
    return cohort_colors

def create_subgroup_colors(subgroup_names, palette='okabe_ito'):
    """Generate consistent color mapping for cohort subgroups.

    Parameters
    ----------
    subgroup_names : list
        List of subgroup identifiers (e.g., ['<40_Male', '<40_Female']).
    palette : str
        RGB palette to use: 'okabe_ito', 'tol_bright', 'tol_muted', etc.

    Returns
    -------
    subgroup_colors : dict
        Mapping of subgroup names to RGB tuples.

    Examples
    --------
    >>> subgroups = ['<40_Male', '<40_Female', '>40_Male', '>40_Female']
    >>> colors = create_subgroup_colors(subgroups)
    >>> # {'<40_Female': (0.0, 0.45, 0.7), ...}
    """
    color_list = create_rgb_palette(palette)

    # Sort subgroup names for deterministic assignments across calls
    sorted_subgroups = sorted(subgroup_names)

    subgroup_colors = {}
    for idx, subgroup in enumerate(sorted_subgroups):
        subgroup_colors[subgroup] = color_list[idx % len(color_list)]

    return subgroup_colors

def create_disease_colors(class_names, palette='okabe_ito'):
    """
    Generate consistent color mapping for disease classes.
    
    Uses semantic color assignments based on medical conventions:
    - Healthy/Control states use cooler colors (blue/green)
    - Inflammatory states use warmer colors (orange/red)
    - Intermediate states use middle colors (yellow)
    
    Parameters
    ----------
    class_names : list
        List of disease class names (e.g., ['HC', 'CD', 'UC', 'SC'])
    palette : str
        RGB palette to use: 'okabe_ito', 'tol_bright', 'tol_muted', etc.
    
    Returns
    -------
    disease_colors : dict
        Mapping of disease names to RGB tuples
    
    Examples
    --------
    >>> classes = ['HC', 'CD', 'UC']
    >>> colors = create_disease_colors(classes)
    >>> # {'HC': (0.0, 0.6, 0.5), 'CD': (0.9, 0.6, 0.0), 'UC': (0.8, 0.4, 0.0)}
    """
    color_list = create_rgb_palette(palette)
    
    # Semantic color mapping based on medical hierarchy
    # These assignments work well across different colorblind-friendly palettes
    if palette == 'okabe_ito':
        # Okabe-Ito palette semantic mapping
        color_mapping = {
            # Healthy/Control - Blue/Green tones
            'HC': color_list[2],        # Bluish green
            'Non-IBD': color_list[2],   # Bluish green
            'Control': color_list[0],   # Blue
            
            # Intermediate/Functional - Darker tones
            'SC': color_list[0],       # Blue

            # Inflammatory - Orange/Red tones
            'CD': color_list[3],        # Yellow (Crohn's Disease)
            'UC': color_list[5],        # Vermillion (Ulcerative Colitis)
            'IBD': color_list[6]       # Reddish purple (Includes both CD and UC)
        }
    
    elif palette == 'tol_bright':
        # Tol Bright palette semantic mapping
        color_mapping = {
            'HC': color_list[0],        # Blue
            'Non-IBD': color_list[2],   # Green
            'Control': color_list[0],   # Blue
            'SC': color_list[5],       # Yellow
            'IBD': color_list[4],       # Orange
            'CD': color_list[1],        # Red
            'UC': color_list[4],        # Orange
            'Unknown': color_list[6],   # Brown/Gray
        }
    
    elif palette == 'tol_muted':
        # Tol Muted palette semantic mapping
        color_mapping = {
            'HC': color_list[2],        # Teal
            'Non-IBD': color_list[3],   # Green
            'Control': color_list[1],   # Cyan
            'SC': color_list[5],       # Sand
            'IBD': color_list[6],       # Rose
            'CD': color_list[7],        # Wine
            'UC': color_list[8],        # Purple
            'Unknown': color_list[0],   # Indigo
        }
    
    elif palette == 'wong':
        # Wong palette semantic mapping
        color_mapping = {
            'HC': color_list[3],        # Bluish green
            'Non-IBD': color_list[3],   # Bluish green
            'Control': color_list[5],   # Blue
            'SC': color_list[4],       # Yellow
            'IBD': color_list[1],       # Orange
            'CD': color_list[6],        # Vermillion
            'UC': color_list[1],        # Orange
            'Unknown': color_list[0],   # Black
        }
    
    else:
        # Generic fallback for other palettes
        color_mapping = {
            'HC': color_list[0],
            'Non-IBD': color_list[0],
            'Control': color_list[0],
            'SC': color_list[1],
            'IBD': color_list[2],
            'CD': color_list[3] if len(color_list) > 3 else color_list[2],
            'UC': color_list[4] if len(color_list) > 4 else color_list[2],
            'Unknown': color_list[-1],
        }
    
    # Create mapping for provided classes
    disease_colors = {}
    for class_name in class_names:
        if class_name in color_mapping:
            disease_colors[class_name] = color_mapping[class_name]
        else:
            # Fallback: assign sequentially based on sorted order
            sorted_classes = sorted(class_names)
            idx = sorted_classes.index(class_name)
            disease_colors[class_name] = color_list[idx % len(color_list)]
    
    return disease_colors


def create_metric_colors(metric_names, palette='okabe_ito'):
    """
    Generate consistent color mapping for evaluation metrics.
    
    Assigns semantically meaningful colors to common ML/statistics metrics:
    - Performance metrics (AUROC, Accuracy) use prominent colors
    - Loss metrics (LogLoss, Brier) use distinct colors
    - Correlation metrics (MCC) use balanced colors
    
    Parameters
    ----------
    metric_names : list
        List of metric names (e.g., ['AUROC', 'MCC', 'LogLoss', 'Brier'])
    palette : str
        RGB palette to use: 'okabe_ito', 'tol_bright', 'tol_muted', etc.
    
    Returns
    -------
    metric_colors : dict
        Mapping of metric names to RGB tuples
    
    Examples
    --------
    >>> metrics = ['AUROC', 'MCC', 'LogLoss', 'Brier']
    >>> colors = create_metric_colors(metrics)
    """
    color_list = create_rgb_palette(palette)
    
    # Semantic color mapping for common metrics
    if palette == 'okabe_ito':
        color_mapping = {
            # Primary performance metrics
            'AUROC': color_list[0],     # Blue
            'AUC': color_list[0],       # Blue
            'Accuracy': color_list[2],  # Bluish green
            'ACC': color_list[2],       # Bluish green
            
            # Correlation/agreement metrics
            'MCC': color_list[1],       # Orange
            'Kappa': color_list[1],     # Orange
            
            # Loss metrics
            'LogLoss': color_list[5],   # Vermillion
            'Brier': color_list[6],     # Reddish purple
            'Loss': color_list[5],      # Vermillion
            
            # Classification metrics
            'F1': color_list[3],        # Yellow
            'Precision': color_list[4], # Sky blue
            'Recall': color_list[6],    # Reddish purple
            'Sensitivity': color_list[6],
            'Specificity': color_list[4],
            
            # Other
            'R2': color_list[0],        # Blue
            'RMSE': color_list[5],      # Vermillion
            'MAE': color_list[5],       # Vermillion
        }
    
    elif palette == 'tol_bright':
        color_mapping = {
            'AUROC': color_list[0],     # Blue
            'AUC': color_list[0],
            'Accuracy': color_list[2],  # Green
            'ACC': color_list[2],
            'MCC': color_list[4],       # Orange
            'Kappa': color_list[4],
            'LogLoss': color_list[1],   # Red
            'Brier': color_list[3],     # Purple
            'Loss': color_list[1],
            'F1': color_list[5],        # Yellow
            'Precision': color_list[0],
            'Recall': color_list[1],
            'Sensitivity': color_list[1],
            'Specificity': color_list[2],
            'R2': color_list[0],
            'RMSE': color_list[1],
            'MAE': color_list[6],       # Brown
        }
    
    elif palette == 'tol_muted':
        color_mapping = {
            'AUROC': color_list[0],     # Indigo
            'AUC': color_list[0],
            'Accuracy': color_list[3],  # Green
            'ACC': color_list[3],
            'MCC': color_list[6],       # Rose
            'Kappa': color_list[6],
            'LogLoss': color_list[7],   # Wine
            'Brier': color_list[8],     # Purple
            'Loss': color_list[7],
            'F1': color_list[5],        # Sand
            'Precision': color_list[1], # Cyan
            'Recall': color_list[2],    # Teal
            'Sensitivity': color_list[2],
            'Specificity': color_list[3],
            'R2': color_list[0],
            'RMSE': color_list[7],
            'MAE': color_list[6],
        }
    
    else:
        # Generic fallback
        color_mapping = {
            'AUROC': color_list[0],
            'AUC': color_list[0],
            'Accuracy': color_list[1],
            'ACC': color_list[1],
            'MCC': color_list[2],
            'Kappa': color_list[2],
            'LogLoss': color_list[3],
            'Brier': color_list[4],
            'Loss': color_list[3],
            'F1': color_list[5] if len(color_list) > 5 else color_list[0],
            'Precision': color_list[0],
            'Recall': color_list[1],
            'Sensitivity': color_list[1],
            'Specificity': color_list[2],
            'R2': color_list[0],
            'RMSE': color_list[3],
            'MAE': color_list[4],
        }
    
    # Create mapping for provided metrics
    metric_colors = {}
    for metric in metric_names:
        if metric in color_mapping:
            metric_colors[metric] = color_mapping[metric]
        else:
            # Fallback: assign sequentially based on sorted order
            sorted_metrics = sorted(metric_names)
            idx = sorted_metrics.index(metric)
            metric_colors[metric] = color_list[idx % len(color_list)]
    
    return metric_colors

def create_model_colors(model_names, palette='okabe_ito'):
    """
    Generate consistent color mapping for machine learning models.

    Parameters
    ----------
    model_names : list
        List of model names (e.g., ['LR', 'SM', 'RF', 'XB]).
    palette : str
        RGB palette to use: 'okabe_ito', 'tol_bright', 'tol_muted', etc.
    Returns
    -------
    model_colors : dict
        Mapping of model names to RGB tuples.
    """
    color_list = create_rgb_palette(palette)
    standard_models = [
        'LR', 'SM', 'NB', 'KNN', 'SVM', 'DT', 'RF', 'GB', 'XB', 'LGBM', 'CAT'
    ]

    if palette == 'okabe_ito':
        color_mapping = {
            'LR': color_list[0],
            'SM': color_list[3],
            'NB': color_list[4],
            'KNN': color_list[2],
            'SVM': color_list[6],
            'DT': color_list[1],
            'RF': color_list[4],
            'GB': color_list[7],
            'XB': color_list[5],
            'LGBM': color_list[3],
            'CAT': color_list[2],
        }
    elif palette == 'tol_bright':
        color_mapping = {
            'LR': color_list[0],
            'SM': color_list[4],
            'NB': color_list[2],
            'KNN': color_list[5],
            'SVM': color_list[3],
            'DT': color_list[1],
            'RF': color_list[6],
            'GB': color_list[5],
            'XB': color_list[2],
            'LGBM': color_list[4],
            'CAT': color_list[1],
        }
    elif palette == 'tol_muted':
        color_mapping = {
            'LR': color_list[0],
            'SM': color_list[5],
            'NB': color_list[2],
            'KNN': color_list[3],
            'SVM': color_list[6],
            'DT': color_list[7],
            'RF': color_list[8],
            'GB': color_list[1],
            'XB': color_list[2],
            'LGBM': color_list[5],
            'CAT': color_list[6],
        }
    else:
        color_mapping = {name: color_list[idx % len(color_list)]
                         for idx, name in enumerate(standard_models)}
        
    model_colors = {}
    sorted_models = sorted(model_names)

    for model in model_names:
        if model in color_mapping:
            model_colors[model] = color_mapping[model]
        else:
            idx = sorted_models.index(model)
            model_colors[model] = color_list[idx % len(color_list)]
    return model_colors

def create_processor_colors(processor_names, palette='okabe_ito'):
    """
    Generate consistent color mapping for preprocessing methods.

    Parameters
    ----------
    processor_names : list
        List of preprocessing method names (e.g., ['Raw', 'CLR', 'ILR']).
    palette : str
        RGB palette to use: 'okabe_ito', 'tol_bright', 'tol_muted', etc.

    Returns
    -------
    processor_colors : dict
        Mapping of preprocessing method names to RGB tuples.

    Examples
    --------
    >>> preprocessors = ['Raw', 'CLR', 'ILR', 'PQN']
    >>> colors = create_processor_colors(preprocessors)
    """
    color_list = create_rgb_palette(palette)

    standard_processors = [
        'Raw', 'Log2', 'PQN', 'CLR', 'ILR', 'Motif', 'GlyCmp',
    ]

    if palette == 'okabe_ito':
        color_mapping = {
            'Raw': color_list[0],
            'Log2': color_list[3],
            'PQN': color_list[4],
            'CLR': color_list[2],
            'ILR': color_list[6],
            'Motif': color_list[1],
            'GlyCmp': color_list[5],
        }

    elif palette == 'tol_bright':
        color_mapping = {
            'Raw': color_list[0],
            'Log2': color_list[4],
            'PQN': color_list[2],
            'CLR': color_list[5],
            'ILR': color_list[3],
            'Motif': color_list[1],
            'GlyCmp': color_list[6],
        }

    elif palette == 'tol_muted':
        color_mapping = {
            'Raw': color_list[0],
            'Log2': color_list[5],
            'PQN': color_list[2],
            'CLR': color_list[3],
            'ILR': color_list[6],
            'Motif': color_list[7],
            'GlyCmp': color_list[8],
        }

    else:
        color_mapping = {name: color_list[idx % len(color_list)]
                         for idx, name in enumerate(standard_processors)}

    processor_colors = {}
    sorted_processors = sorted(processor_names)

    for processor in processor_names:
        if processor in color_mapping:
            processor_colors[processor] = color_mapping[processor]
        else:
            idx = sorted_processors.index(processor)
            processor_colors[processor] = color_list[idx % len(color_list)]

    return processor_colors


def create_color_mapping(names, category='generic', palette='okabe_ito'):
    """
    Universal color mapping generator with consistent assignments.
    
    This function ensures that the same names always get the same colors
    within a given category and palette, regardless of the order they're
    provided in the input list.
    
    Parameters
    ----------
    names : list
        List of names to assign colors to
    category : str
        Category type: 'cohort', 'disease', 'metric', or 'generic'
    palette : str
        RGB palette to use: 'okabe_ito', 'tol_bright', 'tol_muted', etc.
    
    Returns
    -------
    color_mapping : dict
        Mapping of names to RGB tuples
    
    Examples
    --------
    >>> # Cohort colors - always consistent regardless of input order
    >>> colors1 = create_color_mapping(['IT', 'NL', 'US'], category='cohort')
    >>> colors2 = create_color_mapping(['US', 'IT', 'NL'], category='cohort')
    >>> # colors1['IT'] == colors2['IT']  # True!
    >>> 
    >>> # Disease colors - semantic assignments
    >>> disease_colors = create_color_mapping(['HC', 'CD', 'UC'], category='disease')
    >>> 
    >>> # Metric colors - semantic assignments
    >>> metric_colors = create_color_mapping(['AUROC', 'MCC'], category='metric')
    >>> 
    >>> # Generic - sorted alphabetically for consistency
    >>> generic_colors = create_color_mapping(['zebra', 'apple', 'banana'], category='generic')
    """
    if category == 'cohort':
        return create_cohort_colors(names, palette)
    elif category == 'disease':
        return create_disease_colors(names, palette)
    elif category == 'metric':
        return create_metric_colors(names, palette)
    elif category == 'processor':
        return create_processor_colors(names, palette)
    elif category == 'model':
        return create_model_colors(names, palette)
    elif category == 'generic':
        # Generic sequential assignment based on sorted order for consistency
        color_list = create_rgb_palette(palette)
        sorted_names = sorted(names)
        return {name: color_list[sorted_names.index(name) % len(color_list)] 
                for name in names}
    else:
        raise ValueError(
            f"Unknown category: '{category}'. "
            f"Choose from: 'cohort', 'disease', 'metric', 'generic'"
        )

def get_color_for_item(item_name, category='generic', palette='okabe_ito', 
                       known_items=None):
    """
    Get a single color for an item, ensuring consistency with a broader context.
    
    This is useful when you need to get colors one at a time but want them
    to be consistent with a larger set of items.
    
    Parameters
    ----------
    item_name : str
        Name of the item to get color for
    category : str
        Category type: 'cohort', 'disease', 'metric', or 'generic'
    palette : str
        RGB palette to use
    known_items : list, optional
        Full list of items in the dataset for consistent color assignment.
        If None, only the single item is considered.
    
    Returns
    -------
    color : tuple
        RGB tuple for the item
    
    Examples
    --------
    >>> # Get color for a single cohort, consistent with full cohort list
    >>> all_cohorts = ['IT', 'NL', 'UK', 'US']
    >>> color = get_color_for_item('NL', category='cohort', known_items=all_cohorts)
    >>> 
    >>> # This will give the same color as:
    >>> colors = create_color_mapping(all_cohorts, category='cohort')
    >>> colors['NL']  # Same as 'color' above
    """
    if known_items is None:
        known_items = [item_name]
    
    color_mapping = create_color_mapping(known_items, category=category, palette=palette)
    return color_mapping[item_name]

def generate_semantic_colors(palette='okabe_ito', preview=True):
    """
    Create a visual preview of semantic color assignments for a given palette.
    
    Shows how diseases, metrics, and cohorts would be colored using the
    specified palette.
    
    Parameters
    ----------
    palette : str
        RGB palette to use: 'okabe_ito', 'tol_bright', 'tol_muted', etc.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure showing color assignments
    
    Examples
    --------
    >>> fig = generate_semantic_colors('okabe_ito')
    >>> fig.savefig('semantic_colors_preview.png', dpi=300)
    """
    
    # Define test sets
    diseases = ['HC', 'SC', 'Non-IBD', 'CD', 'UC', 'IBD']
    metrics = ['AUROC', 'AUPRC', 'LogLoss', 'Brier']
    cohorts = ['UK', 'US', 'IT', 'NL']
    subgroups = ['<40 | M', '<40 | F', '>40 | M', '>40 | F']
    processors = ['Raw', 'CLR', 'GlyCmp']
    models = ['LR', 'XB']
    
    # Get color mappings
    disease_colors = create_disease_colors(diseases, palette=palette)
    metric_colors = create_metric_colors(metrics, palette=palette)
    cohort_colors = create_cohort_colors(cohorts, palette=palette)
    subgroup_colors = create_subgroup_colors(subgroups, palette=palette)
    processor_colors = create_processor_colors(processors, palette=palette)
    model_colors = create_model_colors(models, palette=palette)
    
    # Create figure
    fig, axes = plt.subplots(6, 1, figsize=(8, 8))
    
    categories = [
        ('Disease Classes', diseases, disease_colors),
        ('Metrics', metrics, metric_colors),
        ('Cohorts', cohorts, cohort_colors),
        ('Processors', processors, processor_colors),
        ('Subgroups', subgroups, subgroup_colors),
        ('Models', models, model_colors),
    ]
    
    for ax, (title, items, colors) in zip(axes, categories):
        # Plot color swatches
        for i, item in enumerate(items):
            color = colors[item]
            ax.add_patch(plt.Rectangle((i, 0), 1, 1, facecolor=color, 
                                       edgecolor='white', linewidth=2))
            # Add label
            ax.text(i + 0.5, 0.5, item, ha='center', va='center', 
                   fontsize=11, fontweight='bold', color='white',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.3))
        
        ax.set_xlim(0, len(items))
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=12, fontweight='bold', loc='left', pad=10)
    
    fig.suptitle(f'Semantic Color Assignments: {palette}', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    if preview:
        plt.show()
    else:
        plt.close(fig)
    
    return disease_colors, cohort_colors, metric_colors, subgroup_colors, processor_colors, model_colors


def validate_color_consistency(names_list1, names_list2, category='generic', 
                                palette='okabe_ito'):
    """
    Validate that color assignments are consistent between two name lists.
    
    Useful for testing that the same items get the same colors regardless
    of the order or context in which they appear.
    
    Parameters
    ----------
    names_list1 : list
        First list of names
    names_list2 : list
        Second list of names (can be different order or subset)
    category : str
        Category type
    palette : str
        RGB palette to use
    
    Returns
    -------
    is_consistent : bool
        True if all common names have the same colors
    differences : dict
        Dictionary of any color differences found
    
    Examples
    --------
    >>> list1 = ['HC', 'CD', 'UC']
    >>> list2 = ['UC', 'HC', 'CD']  # Different order
    >>> is_consistent, diffs = validate_color_consistency(list1, list2, 'disease')
    >>> print(is_consistent)  # Should be True
    """
    colors1 = create_color_mapping(names_list1, category=category, palette=palette)
    colors2 = create_color_mapping(names_list2, category=category, palette=palette)
    
    # Find common names
    common_names = set(names_list1) & set(names_list2)
    
    differences = {}
    for name in common_names:
        if colors1[name] != colors2[name]:
            differences[name] = {
                'list1': colors1[name],
                'list2': colors2[name]
            }
    
    is_consistent = len(differences) == 0
    
    return is_consistent, differences

def rcparams_aga(palette='tol_muted', set_color_cycle=True):
    """
    Configure matplotlib plotting parameters for AGA journal submission.
    
    Complies with AGA requirements:
    - Sans-serif font (Arial/Helvetica), 8-10 point
    - 300 PPI minimum resolution
    - Portrait orientation, max 7" wide x 9" tall
    - CMYK color space
    - Colorblind-friendly colors (default)
    - Output as TIFF, JPEG, EPS, or PDF (not SVG)
    
    Parameters
    ----------
    palette : str
        CMYK palette to use for default color cycle: 'colorblind', 'default', 
        'vibrant', 'muted', 'grayscale'
    set_color_cycle : bool
        If True, sets matplotlib's default color cycle to the chosen palette
    
    Examples
    --------
    >>> # Use colorblind-safe palette (default)
    >>> rcparams_aga()
    >>> 
    >>> # Use vibrant colors
    >>> rcparams_aga(palette='vibrant')
    >>> 
    >>> # Don't set color cycle (manual color control)
    >>> rcparams_aga(set_color_cycle=False)
    """
    mpl.style.use('default')
    
    # Set color cycle if requested
    if set_color_cycle:
        colors_rgb = create_rgb_palette(palette)
        # Convert to hex for matplotlib
        colors_hex = [mpl.colors.rgb2hex(color) for color in colors_rgb]
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors_hex)
    
    plt.rcParams.update({
        # Output format: Use PDF or EPS for vector quality
        'savefig.format': 'pdf',
        
        # Font settings - MUST be sans-serif per requirements
        'font.size': 9,  # Default body text: 8-10 point
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'pdf.fonttype': 42,  # TrueType fonts (editable in Adobe Illustrator)
        'ps.fonttype': 42,
        'text.usetex': False,
        
        # Line and marker styles
        'lines.linewidth': 1.5,
        'lines.markersize': 4,
        
        # Tick parameters
        'xtick.direction': 'out',
        'xtick.top': False,
        'xtick.bottom': True,
        'xtick.minor.visible': False,
        'xtick.labelsize': 9,  # 8-10 point range
        'xtick.minor.size': 2,
        'xtick.minor.width': 0.5,
        'xtick.major.pad': 3,
        'xtick.major.size': 3,
        'xtick.major.width': 1,
        
        'ytick.direction': 'out',
        'ytick.right': False,
        'ytick.left': True,
        'ytick.minor.visible': False,
        'ytick.labelsize': 9,  # 8-10 point range
        'ytick.minor.size': 2,
        'ytick.minor.width': 0.5,
        'ytick.major.pad': 3,
        'ytick.major.size': 3,
        'ytick.major.width': 1,

        # Axes parameters
        'axes.grid': False,
        'axes.edgecolor': 'black',
        'axes.facecolor': 'white',
        'axes.spines.right': False,
        'axes.spines.top': False,
        'axes.titlesize': 9,  # 8-10 point
        'axes.titlepad': 5,
        'axes.labelsize': 9,  # 8-10 point
        'axes.linewidth': 1,
        'axes.labelpad': 3,
        
        # Legend parameters
        'legend.fontsize': 9,
        'legend.frameon': True,
        'legend.framealpha': 1.0,
        'legend.edgecolor': 'black',
        'legend.fancybox': False,
        
        # Figure parameters
        'figure.facecolor': 'white',
        'figure.dpi': 100,  # Screen DPI
        'figure.autolayout': False,
        
        # Save figure parameters - HIGH RESOLUTION
        'savefig.transparent': False,
        'savefig.dpi': 300,  # 300 PPI minimum for all output
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })


def create_figure_with_panels(figsize=(6.5, 8), layout=(2, 1), sharex=False, sharey=False):
    """
    Create a multi-panel figure compliant with AGA requirements.
    
    Parameters
    ----------
    figsize : tuple
        Figure size in inches (width, height). Default (6.5, 8) leaves room for margins.
        Must not exceed (7, 9) as per requirements.
    layout : tuple
        Grid layout as (rows, cols). Default (2, 1) creates 2 panels vertically.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray
        Array of matplotlib.axes.Axes (flattened)
    
    Examples
    --------
    >>> # Vertical 3-panel figure
    >>> fig, axes = create_figure_with_panels(figsize=(6.5, 9), layout=(3, 1))
    >>> 
    >>> # Horizontal 3-panel figure
    >>> fig, axes = create_figure_with_panels(figsize=(6.5, 4), layout=(1, 3))
    >>> 
    >>> # 2x2 grid
    >>> fig, axes = create_figure_with_panels(figsize=(6.5, 8), layout=(2, 2))
    """
    rows, cols = layout
    fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex=sharex, sharey=sharey)
    axes = np.atleast_1d(axes).flatten()
    
    # Add panel labels (16pt Arial bold as required)
    panel_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    for idx, ax in enumerate(axes):
        if idx < len(panel_labels):
            # Position at top-left, outside the plot area
                 ax.text(-0.20, 1.05, panel_labels[idx], 
                   transform=ax.transAxes,
                   fontsize=16, 
                   fontweight='bold',
                   fontfamily='sans-serif',
                   verticalalignment='top',
                   horizontalalignment='right')
    
    return fig, axes

def convert_to_cmyk_tiff(input_file, output_file=None, dpi=300):
    """
    Convert an RGB image to CMYK TIFF format for journal submission.
    
    This should be called AFTER saving your figure with matplotlib.
    
    Parameters
    ----------
    input_file : str or Path
        Path to RGB image (PNG, JPEG, or PDF)
    output_file : str or Path, optional
        Path for CMYK output. If None, creates filename with '_cmyk.tiff' suffix
    dpi : int
        DPI for output (default: 300)
    
    Returns
    -------
    output_file : str
        Path to the saved CMYK TIFF file
    
    Examples
    --------
    >>> fig.savefig('figure1.png', dpi=300)
    >>> convert_to_cmyk_tiff('figure1.png')  # Creates figure1_cmyk.tiff
    """
    
    input_file = Path(input_file)
    
    if output_file is None:
        # Create output filename
        output_file = input_file.with_name(
            input_file.stem + '_cmyk'
        ).with_suffix('.tiff')
    else:
        output_file = Path(output_file)
    
    # Open and prepare image
    img = Image.open(input_file)
    
    # Handle transparency (if present)
    if img.mode == 'RGBA':
        # Create white background
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Convert to CMYK
    img_cmyk = img.convert('CMYK')
    
    # Save as TIFF with compression
    img_cmyk.save(
        output_file, 
        'TIFF', 
        dpi=(dpi, dpi), 
        compression='tiff_lzw'
    )
    
    print(f"✓ CMYK TIFF created")
    return str(output_file)

def save_figure_aga_compliant(fig, filename, dpi=500, format='pdf', 
                               path='.', save_rgb=False, save_cmyk=True):
    """
    Save figure in AGA-compliant format with optional CMYK conversion.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object to save
    filename : str
        Output filename (without extension)
    dpi : int
        Resolution in DPI (300 minimum per requirements)
    format : str
        Output format: 'pdf', 'eps', 'tiff', 'png', or 'jpeg'
    path : str or Path
        Directory path to save the figure
    save_rgb : bool
        If True, keep the RGB version on disk
    save_cmyk : bool
        If True, save the CMYK TIFF version on disk

    Returns
    -------
    rgb_file : str or None
        Path to RGB version if kept
    cmyk_file : str or None
        Path to CMYK version (if save_cmyk=True)
    """
    if dpi < 400:
        print("Warning: DPI is below 400 PPI requirement. Adjusting to 400.")
        dpi = 400
    
    # Ensure figure size complies
    figsize = fig.get_size_inches()
    if figsize[0] > 7 or figsize[1] > 9:
        print(f"Warning: Figure size {figsize} exceeds maximum (7, 9) inches")
    
    # Create path
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    
    # Create RGB version first (required for CMYK conversion)
    rgb_path = path / f"{filename}_rgb.{format}"
    fig.savefig(
        rgb_path,
        dpi=dpi,
        format=format,
        bbox_inches='tight',
        pad_inches=0.05,
        facecolor='white',
        edgecolor='none'
    )

    # Convert RGB to CMYK TIFF
    cmyk_path = path / f"{filename}.tiff"
    convert_to_cmyk_tiff(rgb_path, cmyk_path, dpi=dpi)

    # Keep or delete outputs according to save flags
    if save_rgb:
        rgb_file = str(rgb_path)
        print(f"✓ Saved RGB")
    else:
        if rgb_path.exists():
            rgb_path.unlink()
        rgb_file = None

    if save_cmyk:
        cmyk_file = str(cmyk_path)
    else:
        if cmyk_path.exists():
            cmyk_path.unlink()
        cmyk_file = None

    return rgb_file, cmyk_file