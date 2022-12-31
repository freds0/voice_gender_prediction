from torchmetrics.functional.classification import binary_accuracy, binary_recall, binary_precision, binary_f1_score


def recall(prediction, target):
    return binary_recall(prediction, target)

def precision(prediction, target):
    return binary_precision(prediction, target)

def f1_score(prediction, target):
    return binary_f1_score(prediction, target)

def accuracy(prediction, target):
    return binary_accuracy(prediction, target)

