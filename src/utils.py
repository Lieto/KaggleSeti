import os
import random

from libraries import *


def get_score(y_true, y_pred):

    score = roc_auc_score(y_true, y_pred)
    return score


def seed_torch(seed=42):

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
