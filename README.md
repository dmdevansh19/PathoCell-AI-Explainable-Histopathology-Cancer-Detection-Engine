# PathoCell AI: Explainable Histopathology Cancer Detection Engine

Mini-project (B.Tech CSE AI & ML, 3rd Year, GLA University) by
Devansh, Navneet Soni, and Himanshu Chaudhary.

## Setup (PowerShell)
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Pipeline
1. Download BreakHis into data/raw/ (extract it there)
2. `python scripts/sort_breakhis.py`   -> fills data/processed/benign + malignant
3. `python -m src.train --epochs 3`    -> quick test run
4. `python -m src.train --epochs 15`   -> full run, saves models/best_model.pt
5. `python -m src.evaluate`            -> accuracy / F1 / confusion matrix
6. `streamlit run app/app.py`          -> demo app
