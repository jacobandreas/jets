#!/usr/bin/env python2

from models import *

from collections import namedtuple
import tensorflow as tf

LABELS = {
    "-0.667": 0,
    "-0.333": 1,
    "0.000": 2,
    "0.333": 3,
    "0.667": 4
}

N_BATCH = 100
N_CONST_FEATURES = 4
N_SUMMARY_FEATURES = 13

N_TRAIN = 50000
N_VAL = 1000

N_UPDATE = 100

Datum = namedtuple("Datum", ["label", "summary_features", "const_features"])

def read_data():
    data = []
    max_constituents = 0
    with open("data/data.txt") as data_f:
        c = 0
        for line in data_f:
            c += 1
            if c > 51000:
                break
            parts = line.strip().split()
            #print line
            charge = parts[4]
            label = LABELS[charge]
            bulk_features = np.asarray([float(f) for f in parts[5:6]])
            const_parts = parts[6:]
            assert len(const_parts) % 4 == 0
            const_features = []
            for i_const in range(0, len(const_parts), 4):
                const = const_parts[i_const:i_const+4]
                const_features.append([float(f) for f in const])
            const_features = np.asarray(const_features)

            const_mins = np.min(const_features, axis=0)
            const_maxes = np.max(const_features, axis=0)
            const_means = np.mean(const_features, axis=0)
            summary_features = np.concatenate(
                    (bulk_features, const_mins, const_maxes, const_means))

            data.append(Datum(label, summary_features, const_features))
            max_constituents = max(max_constituents, const_features.shape[0])
            n_summary_features = len(summary_features)

    return data, max_constituents

def do_train_step(model, loader, train_data, session):
    loss, _ = session.run([model.t_loss, model.o_train], loader.load(train_data))
    return loss

def do_val_step(model, loader, train_data, val_data, session):
    train_acc, = session.run([model.t_acc], loader.load(train_data[:N_BATCH], sample=False))
    val_acc = 0
    for i_batch in range(0, N_VAL, N_BATCH):
        val_batch = val_data[i_batch : i_batch + N_BATCH]
        va, = session.run([model.t_acc], loader.load(val_batch, sample=False))
        val_acc += va
    return [train_acc, val_acc / (N_VAL / N_BATCH)]

def main():
    data, max_constituents = read_data()
    train_data = data[:N_TRAIN]
    val_data = data[N_TRAIN:N_TRAIN+N_VAL]

    loader = DataLoader(
            N_BATCH, N_SUMMARY_FEATURES, N_CONST_FEATURES, max_constituents)
    model = MlpModel((len(LABELS),), loader)
    #model = MlpModel((256, 256, len(LABELS),), loader)
    #model = RnnModel(256, len(LABELS), loader)

    session = tf.Session()
    session.run(tf.initialize_all_variables())

    loss = 0
    for t in range(100000):
        loss += do_train_step(model, loader, train_data, session)
        if t > 0 and t % N_UPDATE == 0:
            loss /= N_UPDATE
            acc = do_val_step(model, loader, train_data, val_data, session)
            print loss, acc
            loss = 0

if __name__ == "__main__":
    main()
