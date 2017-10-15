import numpy as np
from skimage.transform import resize
from scipy.ndimage.filters import convolve
from sklearn.svm import LinearSVC

# from time import sleep
# from skimage.io import imread
# import pandas as pd
# from joblib import Parallel, delayed
# from tqdm import tqdm


def brightness_channel(rgb):
    rgb = rgb.astype(np.float16)
    return (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).astype(np.uint8)


def extract_hog(img, cell_rows=8, cell_cols=8, block_row_cells=2, block_col_cells=2, eps=1e-8,
                block_row_stride=2, block_col_stride=2, bins=8):
    img = brightness_channel(img)
    img = resize(img, (32, 32))

    block_rows = block_row_cells * cell_rows
    block_cols = block_col_cells * cell_cols
    Dx = np.array([[-1, 0, 1]])
    Dy = np.array([[-1], [0], [1]])

    Ix = convolve(img, Dx, mode='constant')
    Iy = convolve(img, Dy, mode='constant')
    G = np.sqrt(Ix ** 2 + Iy ** 2)
    # theta
    T = np.arctan2(Iy, Ix)
    T[T < 0] += np.pi
    hog = []

    for block_row_start in range(0, img.shape[0] - block_rows, block_row_stride):
        for block_col_start in range(0, img.shape[1] - block_cols, block_col_stride):

            G_block = G[block_row_start: block_row_start + block_rows,
                      block_col_start: block_col_start + block_cols]
            T_block = T[block_row_start: block_row_start + block_rows,
                      block_col_start: block_col_start + block_cols]
            v = []
            for block_row_cell in range(block_row_cells):
                for block_col_cell in range(block_col_cells):
                    G_cell = G_block[block_row_cell * cell_rows: block_row_cell * cell_rows + cell_rows,
                             block_col_cell * cell_cols: block_col_cell * cell_cols + cell_cols]
                    T_cell = T_block[block_row_cell * cell_rows: block_row_cell * cell_rows + cell_rows,
                             block_col_cell * cell_cols: block_col_cell * cell_cols + cell_cols]
                    hist, _ = np.histogram(T_cell.flatten(), bins, range=(0, np.pi), weights=G_cell.flatten())
                    v.extend(hist)
            v = np.array(v)
            v = v / np.sqrt(np.sum(v ** 2) + eps)
            hog.extend(v)

    return np.array(hog)


def fit_and_classify(X_train, y_train, X_test):
    return LinearSVC(C=0.2).fit(X_train, y_train).predict(X_test)

#
# if __name__ == "__main__":
#     data_train = pd.read_csv("data/00_gt/gt.csv")
#     y_train = data_train.class_id
#     ic_train = [imread("data/00_input/train/" + name) for name in data_train.filename]
#     # X_train = [extract_hog(img) for img in ic_train]
#     X_train = Parallel(n_jobs=8)(delayed(extract_hog)(img) for img in tqdm(ic_train))
#     sleep(10000)
#     fit_and_classify(X_train, y_train, X_train)
