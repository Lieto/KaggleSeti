import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from loguru import logger


def get_train_file_path(data_dir, image_id: str) -> str:

    return data_dir + "/train/{}/{}.npy".format(image_id[0], image_id)


def get_test_file_path(data_dir, image_id) -> str:

    return data_dir + "/test/{}/{}.npy".format(image_id[0], image_id)


def quick_eda(train: pd.DataFrame) -> None:

    plt.figure(figsize=(24, 8))

    for i in range(10):
        image = np.load(train.loc[i, "file_path"])
        image = image.astype(np.float32)
        image = np.vstack(image).transpose((1, 0))

        plt.subplot(5, 2, i + 1)
        plt.imshow(image)

    plt.savefig("QuickEda.png")
    plt.close()

    plt.figure(figsize=(24, 8))

    train["target"].hist()
    plt.savefig("Target_hist.png")
    plt.close()


def load_data(data_dir):

    train: pd.DataFrame = pd.read_csv(data_dir + "/train_labels.csv")
    test: pd.DataFrame = pd.read_csv(data_dir + "/sample_submission.csv")

    train["file_path"] = train["id"].apply(get_train_file_path)
    test["file_path"] = test["id"].apply(get_test_file_path)

    return train, test


def main():

    train, test = load_data()

    logger.info(f"Train head: {train.head()}")
    logger.info(f"Test head: {test.head()}")

    quick_eda(train)


if __name__ == "__main__":
    main()
