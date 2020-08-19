import numpy as np
from tqdm.std import tqdm

from matplotlib import pyplot as plt

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, average_precision_score, precision_recall_curve, auc


def get_performance(y_score, y_true, plot=False):
    best_f1 = 0.0
    best_precision = 0.0
    best_recall = 0.0
    for th in tqdm(np.linspace(np.min(y_score), np.max(y_score), 1000)):
        y_pred = np.zeros(y_score.shape, dtype=np.int64)
        y_pred[y_score < th] = 1
        current_f1 = f1_score(y_true, y_pred)
        current_precision = precision_score(y_true, y_pred)
        current_recall = recall_score(y_true, y_pred)

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_precision = current_precision
            best_recall = current_recall

    score_scaled = (y_score - np.min(y_score)) / (np.max(y_score) - np.min(y_score))
    roc_fprs, roc_tprs, _ = roc_curve(y_true, score_scaled)
    roc_auc = auc(roc_fprs, roc_tprs)
    
    pr_precisions, pr_recalls, _ = precision_recall_curve(y_true, score_scaled)
    # inds = np.argsort(pr_precisions)
    pr_auc = auc(pr_recalls, pr_precisions)

    ap = average_precision_score(y_true, score_scaled)

    roc_fig = None
    pr_fig = None

    if plot:
        roc_fig, roc_ax = plt.subplots()
        roc_ax.plot(roc_fprs, roc_tprs, color='darkorange', lw=1.0, label='ROC Curve (area = %.2f)'%roc_auc)
        roc_ax.plot([0, 1], [0, 1], color='navy', lw=1.0, linestyle='--')
        roc_ax.set_xlim([0.0, 1.0])
        roc_ax.set_ylim([0.0, 1.05])
        roc_ax.set_xlabel('False Positive Rate')
        roc_ax.set_ylabel('True Positive Rate')
        roc_ax.set_title('Receiver operating characteristic curve')
        roc_ax.legend(loc="lower right")

        pr_fig, pr_ax = plt.subplots()
        pr_ax.plot(pr_recalls, pr_precisions, color='darkorange', lw=1.0, label='PR Curve (area = %.2f)' % pr_auc)
        pr_ax.plot([0, 1], [1, 0], color='navy', lw=1.0, linestyle='--')
        pr_ax.set_xlim([0.0, 1.0])
        pr_ax.set_ylim([0.0, 1.05])
        pr_ax.set_xlabel('Precisions')
        pr_ax.set_ylabel('Recalls')
        pr_ax.set_title('Precision recall curve')
        pr_ax.legend(loc="lower right")

    performance = {
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'ap': ap,
        'f1': best_f1,
        'precision': best_precision,
        'recall': best_recall
    }

    return performance, roc_fig, pr_fig
