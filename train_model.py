#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools as it
import argparse as ap
import os, sys, glob
sys.path.append(os.getcwd())
from AE import *
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor

num_marks = 8
num_features = 41

def train_model(X_train, masks_train, X_test, masks_test, batch_size=9000, compress_dim=1500, regularization=1e-3):
    # ae = AE(input_dim=X_train.shape[1],
    #        compress_dim=compress_dim,
    #        batch_size=batch_size,
    #        iterations=200,
    #        regularization=regularization)
    # c, pred, test = ae.train(X_train, masks_train, X_test, masks_test)
    # plt.plot(c)
    ae = MLPRegressor(hidden_layer_sizes=(compress_dim, ), alpha=regularization)
    ae.fit(X_train * masks_train, X_train)
    pred = ae.predict(X_test * masks_test)
    pred_ones = pred[masks_test == 0]
    test_ones = X_test[masks_test == 0]
    r, _ = pearsonr(pred_ones.flatten(), test_ones.flatten())
#     print("Pearson r: %f" % r)
    return r

parser = ap.ArgumentParser()
parser.add_argument('-t', '--training_size', help="Size of training data, default 1e6", nargs='?', type=int, default=100000, const=100000)
parser.add_argument('-f', '--cv_folds', help="Folds for cross-validation, default 5", nargs='?', type=int, default=5, const=5)
parser.add_argument('-m', '--mask_probability', help="Probability that a random mark is masked, default 0.7", nargs='?', type=float, default=0.7, const=0.7)
parser.add_argument('-b', '--batch_size', help="Size for each batch, default 9000", nargs='?', type=int, default=9000, const=9000)
# parser.add_argument('-i', '--impute_file', help="Data file to impute", nargs='?')

args = parser.parse_args()
np.random.seed(0)

train_files = glob.glob("train/full/*.train")
data = []
for f in train_files:
    d = np.loadtxt(f, dtype=np.float32)
    for i in range(d.shape[0]):
        d[i, :] = d[i, :].reshape(-1, num_marks).flatten(order='F')
    data.append(d)

X = np.zeros((data[0].shape[0] * len(data), data[0].shape[1]))
for i in range(len(data)):
    X[i * data[0].shape[0]:(i + 1) * data[0].shape[0]] = data[i]

subset_size = args.training_size
X_subset = X[np.random.randint(0, X.shape[0], subset_size)]
masks = np.random.binomial(1, args.mask_probability, (X_subset.shape[0], num_marks))
masks_full = np.repeat(masks, num_features, axis=1)

# X_train, masks_train = X_subset, masks_full
# test_tracks = np.loadtxt(args.impute_file, dtype=np.float32)

params = {"compress_dim": [1000, 2000, 3000], "regularization": [1e-3, 1e-2, 1e-1]}
param_combinations = list(it.product(*(params[item] for item in sorted(params))))
kf = KFold(n_splits=args.cv_folds)

best_comb = []
best_r = 0.0
best_stdev = 0.0

for comb in param_combinations:
    rs = []
    for train_index, test_index in kf.split(X_subset):
        X_train, masks_train = X_subset[train_index], masks_full[train_index]
        X_test, masks_test = X_subset[test_index], masks_full[test_index]
        r = train_model(X_train, masks_train, X_test, masks_test, batch_size=args.batch_size, compress_dim=comb[0], regularization=comb[1])
        rs.append(r)
    if np.mean(rs) > best_r:
        best_r = np.mean(rs)
        best_stdev = np.std(rs)
        best_comb = comb

print("Best parameters: %d dimensions, %e regularization, r=%.5f(+-%.5f)" % (best_comb[0], best_comb[1], best_r, best_stdev))



