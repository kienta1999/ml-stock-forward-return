#!/usr/bin/env python3
"""Train an XGBoost regressor to predict forward_21d_return.

Tunes max_depth, learning_rate, n_estimators, min_child_weight, subsample
on the validation set. Saves the best model to models/xgb_v1.json.

NOT YET IMPLEMENTED — stub.
"""

import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(_ROOT, "models", "xgb_v1.json")


def train_and_tune() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit("train.py not yet implemented")
