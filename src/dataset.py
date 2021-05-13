from libraries import *
from loguru import logger
from typing import Any
from configuration import CFG

# ==================================
# Dataset
# ==================================


class TrainDataset(Dataset):

    def __init__(self, cfg: CFG, transform: Any = None) -> None:

        self.df: pd.DataFrame = cfg.train_df
        self.file_names = self.df["file_path"].values
        self.labels = self.df[cfg.target_col].values
        self.transform = transform

    def __len__(self) -> int:

        return len(self.df)

    def __getitem__(self, idx: int):

        file_path = self.file_names[idx]
        image = np.load(file_path)
        image = image.astype(np.float32)
        image = np.vstack(image).transpose((1, 0))

        if self.transform:
            image = self.transform(image=image)["image"]
        else:
            image = image[np.newaxis, :, :]
            image = torch.from_numpy(image).float()

        label = torch.tensor(self.labels[idx]).float()

        return image, label


