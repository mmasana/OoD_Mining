import random
import numpy as np
from sklearn import metrics


# Typical metrics, as used in ODIN -- check further at Supplementary Material Section A
def calculate_metrics(y_score, y_true):
    # calculate curves
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score, pos_label=1)
    prec, recall, _ = metrics.precision_recall_curve(y_true, y_score)
    # pick position closer to TPR at 95%
    idx = (np.abs(tpr - 0.95)).argmin()
    # calculate Detection Error
    det_error = 0.5 * (1 - tpr[idx]) + 0.5 * fpr[idx]
    # print the metrics
    print('---')
    print('TPR {} %, FPR {} % '.format(np.round(tpr[idx] * 100, decimals=2), np.round(fpr[idx] * 100, decimals=2)))
    print('Detection Error {} '.format(det_error * 100, decimals=2))
    print('AUROC {} %'.format(np.round(metrics.roc_auc_score(y_true, y_score) * 100, decimals=2)))
    print('AUPR In {} %'.format(np.round(metrics.auc(recall, prec) * 100, decimals=2)))
    # interchange positive and negative classes to print AUPR-out
    y_true = 1 - y_true
    y_score = 1.0 - y_score
    prec, recall, _ = metrics.precision_recall_curve(y_true, y_score)
    print('AUPR Out {} %'.format(np.round(metrics.auc(recall, prec) * 100, decimals=2)))
    # SAFETY CHECK: AUROC should have the same result in both cases
    print('AUROC {} %'.format(np.round(metrics.roc_auc_score(y_true, y_score) * 100, decimals=2)))


# Metrics as used in CC-AG -- check further at Supplementary Material Section A
def calculate_metrics_B(y_score, y_true):
    # Translate to their notation
    idx0 = np.where(y_true == 1)[0]
    idx1 = np.where(y_true == 0)[0]
    X1 = y_score[idx0]
    Y1 = y_score[idx1]
    # Define steps
    end = np.max([np.max(X1), np.max(Y1)])
    start = np.min([np.min(X1), np.min(Y1)])
    gap = (end - start) / 200000
    # TPR95
    total = 0.0
    fpr = 0.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        if tpr <= 0.96 and tpr >= 0.94:
            fpr += error2
            total += 1
    if total == 0:
        print('corner case')
        fprBase = 1
    else:
        fprBase = fpr / total
    # AUROC
    aurocBase = 0.0
    fprTemp = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        fpr = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        aurocBase += (-fpr + fprTemp) * tpr
        fprTemp = fpr
    # AUPRIN
    precisionVec = []
    recallVec = []
    auprBase = 0.0
    recallTemp = 1.0
    for delta in np.arange(start, end, gap):
        tp = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        fp = np.sum(np.sum(Y1 >= delta)) / np.float(len(Y1))
        if tp + fp == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp
        precisionVec.append(precision)
        recallVec.append(recall)
        auprBase += (recallTemp - recall) * precision
        recallTemp = recall
    auprBase += recall * precision
    # AUPROUT
    auprBase2 = 0.0
    recallTemp = 1.0
    for delta in np.arange(end, start, -gap):
        fp = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        tp = np.sum(np.sum(Y1 < delta)) / np.float(len(Y1))
        if tp + fp == 0:
            break
        precision = tp / (tp + fp)
        recall = tp
        auprBase2 += (recallTemp - recall) * precision
        recallTemp = recall
    auprBase2 += recall * precision
    # Detection Error
    errorBase = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        errorBase = np.minimum(errorBase, (tpr + error2) / 2.0)

    # Print metrics
    print('---')
    print('TNR\tDet. Acc.\tAUROC\tAUPR-In\tAUPR-Out')
    print('{}\t{}\t{}\t{}\t{}'.format((1-fprBase)*100, (1-errorBase)*100, aurocBase*100, auprBase*100, auprBase2*100))
    print('---')


def epoch_batches(data_x, data_y, data_out, Params):
    # Batches need to be handled in 4 parts for the combination of pairs (left, right) and distributions (in, out)
    if Params.batch_size % 4 == 0:
        batch_size = Params.batch_size / 2
        quarter_batch = Params.batch_size / 4
    else:
        print("Error: batch size should be a multiple of 4.")
    # Divide data into batches
    batch_count = data_x.shape[0] / batch_size
    num_sim = int(batch_size * Params.siam_batch_ratio)
    b_out_x = np.zeros([batch_count, Params.batch_size, data_x.shape[1], data_x.shape[2], data_x.shape[3]])
    b_out_y = np.zeros([batch_count, Params.batch_size], dtype=int)
    rnd_idx = np.random.permutation(data_x.shape[0])
    b_out_x[:, :batch_size] = np.split(data_x[rnd_idx[:batch_count * batch_size]], batch_count)
    b_out_y[:, :batch_size] = np.split(data_y[rnd_idx[:batch_count * batch_size]], batch_count)
    # Sort data by label
    indx_cls, indx_no_cls = [], []
    for m in range(len(np.unique(data_y))):
        indx_cls.append(np.where(data_y == m)[0])
        indx_no_cls.append(np.where(data_y != m)[0])
    # Generate the pairs
    for b in range(batch_count):
        # Get similar samples
        for m in range(0, num_sim):
            pair = random.sample(indx_cls[b_out_y[b, m]], 1)
            b_out_x[b, m + batch_size] = data_x[pair[0]]
            b_out_y[b, m + batch_size] = data_y[pair[0]]
        # Get dissimilar samples
        for m in range(num_sim, batch_size):
            pair = random.sample(indx_no_cls[b_out_y[b, m]], 1)
            b_out_x[b, m + batch_size] = data_x[pair[0]]
            b_out_y[b, m + batch_size] = data_y[pair[0]]

    # Use unlabelled data from out-of-distribution every other batch
    for b in range(batch_count / 2):
        rnd_out = random.sample(range(len(data_out)), quarter_batch)
        b_out_x[2 * b + 1, 3 * quarter_batch:] = data_out[rnd_out]
        b_out_y[2 * b + 1, 3 * quarter_batch:] = int(Params.num_out)

    return batch_count, b_out_x, b_out_y


def calculate_centers(embeddings, labels):
    num_classes = len(np.unique(labels))
    num_features = embeddings.shape[1]
    cluster_mean = np.zeros([num_classes, num_features])
    for m in range(num_classes):
        indx_cls = np.where(labels == m)[0]
        cluster_mean[m, :] = np.mean(embeddings[indx_cls, :], axis=0)
    return cluster_mean
