import torch


def compute_f1_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    y_true = y_true.long()
    y_pred = y_pred.long()
    f1_scores = []
    for cls in [0, 1]:
        TP = ((y_true == cls) & (y_pred == cls)).sum().item()
        FP = ((y_true != cls) & (y_pred == cls)).sum().item()
        FN = ((y_true == cls) & (y_pred != cls)).sum().item()
        if TP + FP == 0 or TP + FN == 0:
            f1 = 0.0
        else:
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)
        f1_scores.append(f1)
    return sum(f1_scores) / len(f1_scores)
