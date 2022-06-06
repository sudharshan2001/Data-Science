# git clone https://github.com/analokmaus/kuma_utils.git

import pandas as pd
import numpy as np
from tqdm import tqdm

df = pd.read_csv('input csv')

#sklearn iterative imputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


imputer = IterativeImputer(imputation_order='ascending',max_iter=10,random_state=42,n_nearest_features=None)
data = imputer.fit_transform(df)


# LGBM
from kuma_utils.preprocessing.imputer import LGBMImputer

lgbm_imp = LGBMImputer(n_iter=number_of_iterations, verbose=True)
data = lgbm_imp.fit_transform(df)

# KNN
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
knn_imputer = imputer.fit_transform(data)
