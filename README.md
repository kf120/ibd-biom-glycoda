# GlyCoDA - IBD-BIOM

This repository contains code for multiple versions of the manuscript. To reproduce results from a specific version, check out the corresponding tag.

| Version            | Tag                  | Description                                       |
|--------------------|----------------------|---------------------------------------------------|
| medRxiv preprint   | `v1.0-preprint`      | Code accompanying the medRxiv preprint (DOI: 10.64898/2026.04.10.26349930) |
| Revision (current) | `dev` branch         | Active development for journal peer-review        |

## Reproducibility (with uv)

### 1) Environment setup

From the project root:

```bash
uv sync
```

Then start Jupyter (or VS Code notebooks) inside the synced environment.

### 2) Data and paths

- Input `.xlsx` files are expected in the repository-root `datasets/` directory.
- Datasets can be shared upon reasonable request.
- Notebook path handling is centralized in `notebooks/config_paths_ipynb.py`.
- Shared directory constants are defined in `ibd_biom_glycoda/config.py`.
- Figures are written to `figures/`.

### 3) Notebooks

1. `notebooks/01_similarity_analysis.ipynb`
2. `notebooks/02_association_analysis.ipynb`
3. `notebooks/03_glycan_age_analysis.ipynb`
4. `notebooks/04_predictive_modeling_nonibd_ibd.ipynb`
5. `notebooks/05_predictive_modeling_cd_uc.ipynb`

## Repository layout (core)

- `ibd_biom_glycoda/analysis/`: modeling and statistical analysis utilities
- `ibd_biom_glycoda/data/`: data loading and transformations
- `ibd_biom_glycoda/evaluation/`: performance and generalization analysis
- `ibd_biom_glycoda/plotting/`: plotting style and figure export helpers
- `notebooks/`: end-to-end analysis workflows