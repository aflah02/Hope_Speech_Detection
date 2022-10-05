import pandas as pd
from sklearn.model_selection import KFold
import numpy as np

df_english = pd.read_csv('Data Preprocessing\PreprocessedData\english_train_preprocessed.csv')
english_train = df_english.sample(frac=1, random_state=42).reset_index(drop=True)
kf = KFold(n_splits=4)
folds = []
for train_index, test_index in kf.split(english_train):
    X_train, X_test = english_train.iloc[train_index], english_train.iloc[test_index]
    folds.append((X_train, X_test))
folds = np.array(folds)
np.save('Data Preprocessing\PreprocessedData\english_train_folds.npy', folds)