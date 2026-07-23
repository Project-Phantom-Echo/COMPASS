from torch import Tensor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class Metrics:
    def __init__(self):
        self.all_preds = []
        self.all_labels = []

    def update(self, pred: Tensor, target: Tensor):
        self.all_preds.extend(pred.tolist())
        self.all_labels.extend(target.tolist())

    def compute_score(self, prefix=''):
        accuracy = accuracy_score(self.all_labels, self.all_preds)
        precision_micro = precision_score(self.all_labels, self.all_preds, average='micro')
        recall_micro = recall_score(self.all_labels, self.all_preds, average='micro')
        f1_micro = f1_score(self.all_labels, self.all_preds, average='micro')
        precision_macro = precision_score(self.all_labels, self.all_preds, average='macro')
        recall_macro = recall_score(self.all_labels, self.all_preds, average='macro')
        f1_macro = f1_score(self.all_labels, self.all_preds, average='macro')

        return {
            f'{prefix}accuracy': accuracy,
            f'{prefix}precision_micro': precision_micro,
            f'{prefix}recall_micro': recall_micro,
            f'{prefix}f1_micro': f1_micro,
            f'{prefix}precision_macro': precision_macro,
            f'{prefix}recall_macro': recall_macro,
            f'{prefix}f1_macro': f1_macro,
        }

    def reset(self):
        self.all_preds = []
        self.all_labels = []
