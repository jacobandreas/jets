#!/usr/bin/env python2

from models import *

from collections import namedtuple
import cPickle as pickle
import tensorflow as tf
import os

LABELS = {
    "-0.666667": 0,
    "-0.333333": 1,
    "0": 2,
    "0.333333": 3,
    "0.666667": 4
}

REVERSE_LABELS = {v: k for k, v in LABELS.items()}

N_BATCH = 100
N_CONST_FEATURES = 4
N_SUMMARY_FEATURES = 13

N_TRAIN = 600000
N_VAL = 5000

#N_TRAIN = 30000
#N_VAL = 3000

#N_TRAIN = 0
#N_VAL = 1000

N_UPDATE = 10000
#N_UPDATE = 100

#KEEP_CONSTITUENTS = 10
KEEP_CONSTITUENTS = 3

Datum = namedtuple("Datum", ["label", "summary_features", "const_features"])

def read_data():
    if os.path.exists("data/data.p"):
        with open("data/data.p") as pickle_f:
            data, max_constituents = pickle.load(pickle_f)
            return data, max_constituents

    data = []
    max_constituents = 0
    with open("data/data.txt") as data_f:
    #with open("data/check.txt") as data_f:
        c = 0
        for line in data_f:
            c += 1
            parts = line.strip().split()
            charge = parts[4]
            label = LABELS[charge]
            #label = 0
            try:
                bulk_features = np.asarray([float(f) for f in parts[5:6]])
                const_parts = parts[6:]
                assert len(const_parts) % 4 == 0
                const_features = []
                for i_const in range(0, len(const_parts), 4):
                    const = const_parts[i_const:i_const+4]
                    const_features.append([float(f) for f in const])
                const_features = np.asarray(const_features)
            except Exception as e:
                print "WARNING: skipped due to", e
                continue

            const_mins = np.min(const_features, axis=0)
            const_maxes = np.max(const_features, axis=0)
            const_means = np.mean(const_features, axis=0)
            #summary_features = np.concatenate(
            #        (bulk_features, const_mins, const_maxes, const_means))
            summary_features = bulk_features
            #summary_features = np.asarray([0])

            data.append(Datum(label, summary_features, const_features))
            max_constituents = max(max_constituents, const_features.shape[0])
            n_summary_features = len(summary_features)

    np.random.shuffle(data)

    with open("data/data.p", "wb") as pickle_f:
        pickle.dump((data, max_constituents), pickle_f) 

    return data, max_constituents

def do_train_step(model, loader, train_data, session):
    loss, _ = session.run([model.t_loss, model.o_train], loader.load(train_data))
    return loss

def do_val_step(model, loader, train_data, val_data, session):
    train_acc, = session.run([model.t_acc], loader.load(train_data[:N_BATCH], sample=False))
    val_acc = 0
    for i_batch in range(0, N_VAL, N_BATCH):
        val_batch = val_data[i_batch : i_batch + N_BATCH]
        scores, = session.run([model.t_scores], loader.load(val_batch, sample=False))
        preds = np.argmax(scores, axis=1)
        va = np.sum(preds == [datum.label for datum in val_batch])
        val_acc += va
    return [train_acc, 1. * val_acc / N_VAL]

def order_constituents(data):
    for j in range(len(data)):
        datum = data[j]
        feats = datum.const_features
        consts = [feats[i, :] for i in range(feats.shape[0])]
        by_energy = sorted(consts, key=lambda x: x[0])
        by_energy = by_energy[:KEEP_CONSTITUENTS]
        data[j] = datum._replace(const_features=np.asarray(by_energy))

def main():
    data, max_constituents = read_data()
    order_constituents(data)
    max_constituents = KEEP_CONSTITUENTS
    train_data = data[:N_TRAIN]
    val_data = data[N_TRAIN:N_TRAIN+N_VAL]
    #for i_val, datum in enumerate(val_data):
    #    for i_train, train_datum in enumerate(train_data):
    #        if not (not (datum.summary_features == train_datum.summary_features).all() 
    #                and not (datum.const_features == train_datum.const_features).all()): 
    #            print (i_val, i_train)

    loader = DataLoader(
            N_BATCH, N_SUMMARY_FEATURES, N_CONST_FEATURES, max_constituents)
    #model = MlpModel((len(LABELS),), loader)
    #model = MlpModel((256, len(LABELS),), loader)
    model = RnnModel(256, len(LABELS), loader)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    loss = 0
    for t in range(1000000):
        loss += do_train_step(model, loader, train_data, session)
        if t > 0 and t % N_UPDATE == 0:
            loss /= N_UPDATE
            acc = do_val_step(model, loader, train_data, val_data, session)
            print loss, acc
            loss = 0
            model.save(saver, session)

if __name__ == "__main__":
    main()
