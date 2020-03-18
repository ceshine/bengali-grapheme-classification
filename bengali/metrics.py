import numpy as np
import sklearn.metrics

from tf_helper_bot import Metric

from .dataset import SPLIT_POINT


class MacroAveragedRecall(Metric):
    name = "ma_recall"

    def __call__(self, truth: np.ndarray, pred: np.ndarray):
        scores = []
        for i in range(3):
            y_true_subset = truth[:, i]
            y_pred_subset = np.argmax(
                pred[:, SPLIT_POINT[i]:SPLIT_POINT[i+1]],
                axis=1
            )
            scores.append(sklearn.metrics.recall_score(
                y_true_subset, y_pred_subset, average='macro'))
        final_score = np.average(scores, weights=[2, 1, 1])
        formatted = (
            f"{final_score * 100:.2f} | root: {scores[0] * 100:.2f} "
            f"vowel: {scores[1] * 100:.2f}  consonant: {scores[2] * 100:.2f}"
        )
        return final_score * -1, formatted
