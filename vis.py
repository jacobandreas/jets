#!/usr/bin/env python2

from models import *
from main import read_data, Datum, order_constituents
from main import N_TRAIN, N_VAL, N_BATCH, N_SUMMARY_FEATURES, \
        N_CONST_FEATURES, LABELS, REVERSE_LABELS, KEEP_CONSTITUENTS

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import sys

def vis(model, data, loader, session):
    confusions = np.zeros((len(LABELS), len(LABELS)))

    hist = np.zeros((10, len(LABELS)))
    correct = 0
    count = 0

    for i_batch in range(0, len(data), N_BATCH):
        batch = data[i_batch : i_batch + N_BATCH]
        if len(batch) < N_BATCH:
            break
        probs, = session.run([model.t_probs], loader.load(batch, sample=False))
        for i in range(len(batch)):
            confusions[batch[i].label, np.argmax(probs[i, :])] += 1
            if batch[i].label == np.argmax(probs[i, :]):
                correct += 1
            count += 1

            jc = batch[i].summary_features[0]
            bn = int((jc + 1) * 5)
            bn = max(bn, 0)
            bn = min(bn, 9)
            #hist[bn, batch[i].label] += 1
            for label in range(len(LABELS)):
                hist[bn, label] += probs[i, label]

    print confusions
    print 1. * correct / count

    local_norm = hist.copy()
    local_norm /= local_norm.sum(axis=0)[np.newaxis, :]

    global_norm = hist.copy()
    global_norm /= global_norm.sum(axis=1)[:, np.newaxis]

    for i in range(len(LABELS)):
        plt.plot(local_norm[:, i], label=REVERSE_LABELS[i])
    plt.legend()
    plt.savefig("local_norm_p.png")

    plt.clf()
    for i in range(len(LABELS)):
        plt.bar(np.arange(10),
                global_norm[:, i],
                bottom=global_norm[:, :i].sum(axis=1),
                label=REVERSE_LABELS[i])
    plt.legend()
    plt.savefig("global_norm_p.png")

def main():
    data, max_constituents = read_data()
    order_constituents(data)
    max_constituents = KEEP_CONSTITUENTS
    train_data = data[:N_TRAIN]
    val_data = data[N_TRAIN:N_TRAIN+N_VAL]

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
    elif sys.argv[1] == "rnn":
        model = RnnModel(256, len(LABELS), loader)
        path = "save/rnn.chk"
    else:
        assert False

    session = tf.Session()
    saver = tf.train.Saver()
    saver.restore(session, path)

    vis(model, val_data, loader, session)

if __name__ == "__main__":
    main()
