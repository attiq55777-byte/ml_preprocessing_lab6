# ML Preprocessing Lab

Data preprocessing pipeline for the **Bank Customer Churn** dataset (Programming for AI, Lab 6).

## Steps covered
1. Feature selection (correlation heatmap, |r| > 0.1 filter, dropping redundant ID columns)
2. Outlier detection and removal (IQR method, optional PyOD)
3. Imbalanced data handling (class distribution, upsampling, downsampling)
4. Categorical encoding (one-hot encoding of `Geography`, `Gender`)
5. Train / validation / test split (60 / 20 / 20, stratified)

## Structure
```
dataset/bank_churn.csv
notebooks/preprocessing_lab.ipynb
scripts/preprocessing.py
README.md
```

## Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn pyod
python scripts/preprocessing.py          # from the project root
jupyter notebook notebooks/preprocessing_lab.ipynb
```
