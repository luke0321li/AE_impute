#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools as it
import argparse as ap
import tensorflow as tf
import os, sys, glob
sys.path.append(os.getcwd())
from AE import *
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor

num_marks = 8
num_features = 41

def train_model(X_train, masks_train, X_test, masks_test, batch_size=9000, compress_dim=1500, regularization=1e-3):
    ae = AE(input_dim=X_train.shape[1],
            compress_dim=compress_dim,
            batch_size=batch_size,
            iterations=200,
            regularization=regularization)
    c, pred, test = ae.train(X_train, masks_train, X_test, masks_test)
    plt.plot(c)
    pred_ones = pred[masks_test == 1]
    test_ones = test[masks_test == 1]
    r, _ = pearsonr(pred_ones.flatten(), test_ones.flatten())
#     print("Pearson r: %f" % r)
    return r

parser = ap.ArgumentParser()
parser.add_argument('-t', '--training_size', help="Size of training data, default 1e6", nargs='?', type=int, default=100000, const=100000)
# parser.add_argument('-f', '--cv_folds', help="Folds for cross-validation, default 5", nargs='?', type=int, default=5, const=5)
parser.add_argument('-m', '--mask_probability', help="Probability that a random mark is masked, default 0.7", nargs='?', type=float, default=0.7, const=0.7)
parser.add_argument('-b', '--batch_size', help="Size for each batch, default 9000", nargs='?', type=int, default=9000, const=9000)
parser.add_argument('-i', '--impute_file', help="Data file to impute", nargs='?')

args = parser.parse_args()
np.random.seed(0)

train_files = glob.glob("train/full/*.train")
data = []
for f in train_files:
    d = np.loadtxt(f, dtype=np.float32)
    for i in range(d.shape[0]):
        d[i, :] = d[i, :].reshape(-1, 8).flatten(order='F')
    data.append(d)

X = np.zeros((data[0].shape[0] * len(data), data[0].shape[1]))
for i in range(len(data)):
    X[i * data[0].shape[0]:(i + 1) * data[0].shape[0]] = data[i]

subset_size = args.training_size
X_subset = X[np.random.randint(0, X.shape[0], subset_size)]
masks = np.random.binomial(1, args.mask_probability, (X_subset.shape[0], num_marks))
masks_full = np.repeat(masks, num_features, axis=1)
X_train, masks_train = X_subset, masks_full

compress_dim = 2000
regularization = 1e-3
ae = AE(input_dim=X_train.shape[1], compress_dim=compress_dim, batch_size=args.batch_size, iterations=200, regularization=regularization)
# ae = MLPRegressor(hidden_layer_sizes=(compress_dim, ), alpha=regularization)
# ae.fit(X_train * masks_train, X_train)
# c, _, _ = ae.train(X_train, masks_train, X_train, masks_train)

test_tracks = np.loadtxt(args.impute_file, dtype=np.float32)
with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, "./model_1")
    output = np.zeros(test_tracks.shape, dtype=np.float32)
    X_test = np.zeros((len(test_tracks) - num_features, num_features * num_marks))

    for i in range(X_test.shape[0]):
        X_test[i, :] = test_tracks[i:i + num_features, :].flatten(order='F')
        X_test_instance = X_test[i, :].reshape(1, -1)
        output[i:i + num_features, :] += ae.output.eval(feed_dict={ae.x: X_test_instance, ae.mask: np.ones(X_test_instance.shape)}).reshape(num_features, num_marks, order='F')
#             ae.test(X_test_instance, ).reshape(num_features, num_marks, order='F')

    output = output / num_features
    np.savetxt(args.impute_file.split('/')[-1] + ".imputed", output)
