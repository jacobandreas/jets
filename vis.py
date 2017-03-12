#!/usr/bin/env python2

from models import *
from main import read_data
from main import N_VAL, N_BATCH, N_SUMMARY_FEATURES, N_CONST_FEATURES, LABELS, REVERSE_LABELS

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import sys

def vis(model, data, loader, session):
    confusions = np.zeros((len(LABELS), len(LABELS)))

    hist = np.zeros((10, len(LABELS)))

    for i_batch in range(0, len(data), N_BATCH):
        batch = data[i_batch : i_batch + N_BATCH]
        if len(batch) < N_BATCH:
            break
        scores, = session.run([model.t_scores], loader.load(batch, sample=False))
        for i in range(len(batch)):
            confusions[batch[i].label, np.argmax(scores[i, :])] += 1

            jc = batch[i].summary_features[0]
            bn = int((jc + 1) * 5)
            bn = max(bn, 0)
            bn = min(bn, 9)
            hist[bn, batch[i].label] += 1

    #local_norm = hist.copy()
    #local_norm /= local_norm.sum(axis=0)[np.newaxis, :]

    global_norm = hist.copy()
    global_norm /= global_norm.sum(axis=1)[:, np.newaxis]

    for i in range(len(LABELS)):
        plt.plot(global_norm[:, i], label=REVERSE_LABELS[i])
    plt.legend()
    plt.savefig("global_norm.png")

def main():
    data, max_constituents = read_data()
    val_data = data[-N_VAL:]

    #counts = np.zeros(5)
    #for datum in data:
    #    counts[datum.label] += 1
    #counts /= counts.sum()
    #print counts
    #exit()

    loader = DataLoader(
            N_BATCH, N_SUMMARY_FEATURES, N_CONST_FEATURES, max_constituents)

    if sys.argv[1] == "mlp":
        model = MlpModel((len(LABELS),), loader)
        path = "save/mlp.chk"
    else:
        assert False

    session = tf.Session()
    saver = tf.train.Saver()
    saver.restore(session, path)

    vis(model, data, loader, session)

if __name__ == "__main__":
    main()
