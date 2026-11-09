def precision(preds, targets):
    pred = preds.argmax(dim=1)

    pred_lane = pred != 0
    true_lane = targets != 0

    TP = (pred_lane & true_lane).sum().float()
    FP = (pred_lane & ~true_lane).sum().float()

    return TP / (TP + FP + 1e-8)

def recall(preds, targets):
    pred = preds.argmax(dim=1)

    pred_lane = pred != 0
    true_lane = targets != 0    

    TP = (pred_lane & true_lane).sum().float()
    FN = (~pred_lane & true_lane).sum().float()

    return TP / (TP + FN + 1e-8)

def left_precision(preds, targets):
    TP, FP, FN = class_stats(preds, targets, 1)
    return TP / (TP + FP + 1e-8)

def left_recall(preds, targets):
    TP, FP, FN = class_stats(preds, targets, 1)
    return TP / (TP + FN + 1e-8)

def right_precision(preds, targets):
    TP, FP, FN = class_stats(preds, targets, 2)
    return TP / (TP + FP + 1e-8)

def right_recall(preds, targets):
    TP, FP, FN = class_stats(preds, targets, 2)
    return TP / (TP + FN + 1e-8)

def class_stats(preds, targets, cls):
    pred = preds.argmax(dim=1)

    pred_cls = pred == cls
    true_cls = targets == cls

    TP = (pred_cls & true_cls).sum().float()
    FP = (pred_cls & ~true_cls).sum().float()
    FN = (~pred_cls & true_cls).sum().float()

    return TP, FP, FN
