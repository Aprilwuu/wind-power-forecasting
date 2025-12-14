# Probabilistic Wind Power Forecasting

This repository contains code for my MSc research project on probabilistic forecasting of wind power.

## Objectives
- Model wind power uncertainty
- Compare probabilistic forecasting methods
- Evaluate using CRPS, Pinball loss, PICP, coverage, etc.

## Repo Structure
src/         → Python modules  
notebooks/   → Jupyter experiments  
configs/     → Training configs  
scripts/     → Helper scripts  
data/        → (ignored) 

##  Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Evaluation Metrics
	•	CRPS
	•	Pinball Loss
	•	PICP
	•	Interval Width

