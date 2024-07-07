from .Loss import LossBase
from numpy import sum, clip, log


class Loss_CategoricalCrossEntropy(LossBase):
    def forward(self, y_pred, y_true):
        y_pred_clipped = clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(len(y_pred)), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = sum(y_pred_clipped * y_true, axis=1)
        return -log(correct_confidences)
