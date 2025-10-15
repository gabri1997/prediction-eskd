import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score,f1_score,roc_curve,precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Activation, Dropout,ELU
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.optimizers.schedules import ExponentialDecay
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

def preprocess_data(df):

    df = pd.read_csv("/work/grana_far2023_fomo/ESKD/Data/final_merge_with_Year_last.csv")
    print(df.head())
    #1. Hypertension (from BP) 
    df["Hypertension"] = ((df["systolic"] >= 140) | (df["Diastolic"] >= 90)).astype(int)

    #2. Therapy (combine 4 therapy attributes)
    therapy_cols = ["RAS blockers", "Immunotherapies", "fish oil", "Tonsillectomy"]

    # Check for existence of the columns
    existing_therapy_cols = [c for c in therapy_cols if c in df.columns]
    if not existing_therapy_cols:
        raise ValueError(f"None of these therapy columns found: {therapy_cols}")

    # Make sure these columns are numeric (0/1)
    df[existing_therapy_cols] = df[existing_therapy_cols].fillna(0).astype(int)

    # Create Therapy = 1 if any therapy type == 1, else 0
    df["Therapy"] = (df[existing_therapy_cols].sum(axis=1) > 0).astype(int)

    # Define baseline features (from paper, excluding BP)
    baseline_features = [
        "age", "SEX", "systolic", "Diastolic","creat_mg_dl", "Uprot",
        "M", "E", "S", "T", "C",
        "Therapy", "Hypertension"
    ]

    # Keep followup_years + outcome separately
    needed_columns = baseline_features + ["years_since_biopsy", "outcome"]

    # 1. Drop rows with missing values in required features
    df_clean = df.dropna(subset=needed_columns).copy()
    df_clean = df[needed_columns].dropna().copy()
    # 2. Encode sex (M=1, F=0)
    df_clean["SEX"] = df_clean["SEX"].map({"M": 1, "F": 0})
    df_clean.head(10)

    return df_clean, baseline_features

