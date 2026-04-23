# Probabilistic Wind Power Forecasting

This repository contains the code for my MSc research project on probabilistic wind power forecasting.  
The project investigates how different model backbones and uncertainty formulations affect forecasting performance under both temporal and cross-site generalization settings.

## Overview

Wind power is highly variable and uncertain, which makes point forecasting alone insufficient for many real-world applications.  
This project focuses on probabilistic forecasting, with an emphasis on prediction interval quality, calibration, and robustness across different evaluation scenarios.

## Methods

The main forecasting pipelines in this project include:

- **LightGBM + Quantile Regression**
- **Transformer + Quantile Regression**
- **Transformer + Beta Distribution**
- **TCN + MC Dropout + Noise Augmentation**

To improve interval reliability, **split conformal calibration** is applied as a post-processing step.

## Experimental Settings

Two evaluation settings are considered:

- **Track 1: Temporal Generalization**  
  Training and testing are conducted on different time periods within the same wind farms.

- **Track 2: Cross-Site Generalization (LOFO)**  
  Leave-one-farm-out evaluation is used to assess robustness under spatial distribution shift.

## Evaluation Metrics

The models are evaluated using both point prediction and probabilistic forecasting metrics, including:

### Point Forecasting Metrics
- **RMSE**
- **MAE**
- **R²**

### Probabilistic Forecasting Metrics
- **PICP** (Prediction Interval Coverage Probability)
- **MPIW** (Mean Prediction Interval Width)
- **WIS** (Weighted Interval Score)

## Repository Structure

```text
src/         Python modules for data processing, feature engineering, modeling, and post-processing
configs/     YAML configuration files for experiments
scripts/     Experiment entry points and helper scripts
data/        Dataset directory (not included in the repository)
notebooks/   Jupyter notebooks for exploratory analysis and experiments
```

## Data Preparation
The raw dataset files should be placed under:

```text
data/raw/
```
After that, merge the zone-level raw files into one processed dataset:

```bash
python scripts/data/wind_all_zones.py
```

This will generate:

```text
data/processed/gefcom_wind_all_zones.csv
```

## Running Experiments

Experiments are controlled by YAML configuration files in the configs/ directory.

A typical training run can be launched with:

```bash
python scripts/run_pipeline.py
```

Model outputs, metrics, and checkpoints will be saved under the corresponding experiment directory.

## Conformal Calibration

Post-hoc conformal calibration can be applied after model inference to improve interval reliability.

Example:

```bash
python -m scripts.postprocess.run_conformal_and_collect \
    --root data/featured/<experiment_name>
```

## Result Summarization
To summarize results across seeds or held-out sites, use:

```bash
python -m scripts.postprocess.summarize_conformal_ci95 \
    --runs_csv data/featured/<experiment_name>/postprocess_conformal_summary.json
```

This script generates summary tables such as mean, standard deviation, and confidence intervals for the main probabilistic metrics.

## Notes

The dataset target is normalized to [0, 1]
Missing values are handled during preprocessing
No explicit outlier removal is applied
Conformal calibration is applied as a post-processing step rather than being built into the predictive models

## Reproducibility

The codebase is configuration-driven
Experiments are organized by model, track, and seed
Random seeds are fixed where applicable