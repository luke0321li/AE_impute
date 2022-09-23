#!/usr/bin/env python

import sys, gzip
import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor

SAMPLES = []
MARKS = []
SMM = []
GENOME_LEN = 1925196

def read_datafile(sample, mark):
    filename = "chr21_%s-%s.pval.signal.bedGraph.wig.gz" % (sample, mark)
    datapath = "/u/home/l/luke0321/nobackup-WGSPD/roadmap/chr21_primary/"
    try:
        with gzip.open(datapath + filename) as file:
            data = np.zeros(GENOME_LEN, dtype=np.float32)
            line = file.readline()
            line = file.readline()
            line = file.readline()
            index = 0
            while line:
                score = float(line.split()[0])
                data[index] = score
                index += 1
                line = file.readline()
            sys.stderr.write("Finished reading file: %s\n" % (datapath + filename))
            return data
    except FileNotFoundError:
        sys.stderr.write("Error: data file not found\n")
        return 0


def read_sample_mark_matrix():
    sample_marks = {} # Key: sample, Value: list of available marks
    samples = []
    marks = []
    with open("/u/home/l/luke0321/nobackup-WGSPD/roadmap/chr21_primary/tier1_samplemarktable.txt") as file:
        for line in file:
            text = line.split()
            if text[0] not in sample_marks:
                sample_marks[text[0]] = []
                samples.append(text[0])
            sample_marks[text[0]].append(text[1])
            if text[1] not in marks:
                marks.append(text[1])
    # A binary matrix indicating availability of (sample, mark) track
    sample_mark_matrix = np.zeros((len(sample_marks), len(marks)))

    for i in range(len(samples)):
        m = sample_marks[samples[i]]
        for j in range(len(marks)):
            if marks[j] in m:
                sample_mark_matrix[i, j] = 1

    return samples, marks, sample_mark_matrix


def generate_one_dataset(sample, SAMPLES, MARKS, SMM, size=10000, flank=20, mode="train"):
    sample_index = SAMPLES.index(sample)
    data = np.zeros((GENOME_LEN, len(MARKS)))
    for i in range(len(MARKS)):
        data[:, i] = read_datafile(sample, MARKS[i])
    if mode == "train":
        rand_pos = np.random.randint(flank, GENOME_LEN - flank - 1, size=size)
    elif mode == "test":
        # rand_pos = np.arange(flank, GENOME_LEN - flank - 1)
        np.savetxt("%s.%s" % (sample, mode), data)
        return data
    tracks = np.zeros((len(rand_pos), data.shape[1] * (flank * 2 + 1)))
    for i in range(len(rand_pos)):
        tracks[i, :] = data[rand_pos[i] - flank:rand_pos[i] + flank + 1].flatten(order='F')
    np.savetxt("%s.%s" % (sample, mode), tracks)
    return tracks


if __name__ == "__main__":
    s, m, smm = read_sample_mark_matrix()
    for sample in s:
        generate_one_dataset(sample, s, m, smm, mode="train")