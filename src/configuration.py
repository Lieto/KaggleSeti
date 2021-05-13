from libraries import *
from loguru import logger


class CFG:

    apex = False
    debug = False
    print_freq = 100
    num_workers = 4
    model_name = "nfnet_l0"
    size = 224
    scheduler = "CosineAnnealingLR"
    epochs = 6
    T_max = 6
    lr = 1e-4
    min_lr = 1e-6
    batch_size = 64
    weight_decay = 1e-6
    gradient_accumulation_steps = 1
    max_grad_norm = 1000
    seed = 42
    target_size = 1
    target_col = "target"
    n_fold = 4
    trn_fold = [0, 1, 2, 3]
    train = True

    def __init__(self, train_df: pd.DataFrame, debug: bool = False, ) -> None:

        self.debug = debug

        if self.debug:
            self.train_df = train_df.sample(n=1000, random_state=self.seed).reset_index(drop=True)
        else:
            self.train_df = train_df.reset_index(drop=True)

    def cv_split(self):

        fold = StratifiedKFold(n_splits=self.n_fold, shuffle=True, random_state=self.seed)

        for n, (train_index, val_index) in enumerate(fold.split(self.train_df, self.train_df[self.target_col])):
            self.train_df.loc[val_index, "fold"] = int(n)

        self.train_df["fold"] = self.train_df["fold"].astype(int)

        logger.info(f"Train group by fold target: {self.train_df.groupby(['fold', 'target']).size()}")

    def get_transforms(self, *, data):

        if data == "train":

            return A.Compose([
                A.Resize(self.size, self.size),
                ToTensorV2(),
            ])

        elif data == "valid":

            return A.Compose([
                A.Resize(self.size, self.size),
                ToTensorV2(),
            ])


