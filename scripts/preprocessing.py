"""
Bank Customer Churn - ML preprocessing pipeline (Lab 6)
Run:  python scripts/preprocessing.py
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

DATA_PATH = "dataset/bank_churn.csv"
TARGET = "Exited"
RANDOM_STATE = 42
RENAME = {"customer_id": "CustomerId", "credit_score": "CreditScore", "country": "Geography",
          "gender": "Gender", "age": "Age", "tenure": "Tenure", "balance": "Balance",
          "products_number": "NumOfProducts", "credit_card": "HasCrCard",
          "active_member": "IsActiveMember", "estimated_salary": "EstimatedSalary", "churn": "Exited"}


def load_data(path=DATA_PATH):
    # file uses snake_case names; rename to the lab's column names
    return pd.read_csv(path).rename(columns=RENAME)


def select_features(df, threshold=0.1):
    """Features whose correlation with the target is > threshold or < -threshold."""
    corr = df.corr(numeric_only=True)[TARGET].drop(TARGET)
    return corr[(corr > threshold) | (corr < -threshold)]


def drop_redundant(df, cols=("RowNumber", "CustomerId", "Surname")):
    """Drop identifier columns (no predictive value -> pure noise)."""
    return df.drop(columns=[c for c in cols if c in df.columns])


def remove_outliers_iqr(df, cols, k=1.5):
    """Remove rows lying outside [Q1 - k*IQR, Q3 + k*IQR] for any column in cols."""
    mask = pd.Series(True, index=df.index)
    for c in cols:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        mask &= df[c].between(q1 - k * iqr, q3 + k * iqr)
    return df[mask]


def encode_categoricals(df, cols=("Geography", "Gender")):
    return pd.get_dummies(df, columns=list(cols), drop_first=True)


def upsample(features, target, repeat):
    features_0 = features[target == 0]
    features_1 = features[target == 1]
    target_0 = target[target == 0]
    target_1 = target[target == 1]
    features_up = pd.concat([features_0] + [features_1] * repeat)
    target_up = pd.concat([target_0] + [target_1] * repeat)
    return shuffle(features_up, target_up, random_state=RANDOM_STATE)


def downsample(features, target, fraction):
    """Keep only `fraction` of the majority class (0), keep all of class 1."""
    features_0 = features[target == 0]
    features_1 = features[target == 1]
    target_0 = target[target == 0]
    target_1 = target[target == 1]
    features_down = pd.concat([features_0.sample(frac=fraction, random_state=RANDOM_STATE), features_1])
    target_down = pd.concat([target_0.sample(frac=fraction, random_state=RANDOM_STATE), target_1])
    return shuffle(features_down, target_down, random_state=RANDOM_STATE)


def split_data(df):
    """60% train / 20% validation / 20% test (stratified)."""
    train_valid, test = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE, stratify=df[TARGET])
    train, valid = train_test_split(train_valid, test_size=0.25, random_state=RANDOM_STATE,
                                    stratify=train_valid[TARGET])
    out = {}
    for name, part in (("train", train), ("valid", valid), ("test", test)):
        out[name] = (part.drop(TARGET, axis=1), part[TARGET])
    return out


if __name__ == "__main__":
    df = load_data()
    print("Raw shape:", df.shape)
    print("Selected features:\n", select_features(df))

    df = drop_redundant(df)
    df = remove_outliers_iqr(df, ["CreditScore", "Age"])
    df = encode_categoricals(df)
    print("After cleaning/encoding:", df.shape)

    # Split FIRST, then balance only the training set (avoids data leakage)
    sets = split_data(df)
    X_train, y_train = sets["train"]
    X_train, y_train = upsample(X_train, y_train, 3)

    print("Training set  :", X_train.shape, y_train.shape)
    print("Validation set:", sets["valid"][0].shape, sets["valid"][1].shape)
    print("Test set      :", sets["test"][0].shape, sets["test"][1].shape)
    print("Train class distribution:\n", y_train.value_counts(normalize=True) * 100)
