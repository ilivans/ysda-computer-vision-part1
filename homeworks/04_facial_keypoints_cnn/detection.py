from os.path import join, dirname, abspath
from random import shuffle

from copy import deepcopy
from skimage import transform

from keras.layers import Convolution2D, Activation, MaxPooling2D, Dropout, Flatten, Dense, GlobalMaxPooling2D, \
    BatchNormalization
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop, Nadam, SGD, Adam
from keras.losses import mean_squared_error
from numpy import array
from skimage.transform import resize

from os import listdir
from skimage.color import gray2rgb
from skimage.io import imread
import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from subprocess import call


def read_csv(filename):
    res = {}
    with open(filename) as fhandle:
        next(fhandle)
        for line in fhandle:
            parts = line.rstrip('\n').split(',')
            coords = array([float(x) for x in parts[1:]], dtype='float64')
            res[parts[0]] = coords
    return res

NUM_POINTS = 14 * 2
INPUT_SIZE = 100
LEARNING_RATE = 1e-3
MOMENTUM = 0.9
LR_FACTOR = 0.1
EPOCHS = 99
BATCH_SIZE = 512


def random_horizontal_flip(img, pts, u=0.5):
    if np.random.random() < u:
        img = img[:, ::-1]
        pts[::2] = 1-pts[::2]
    return img, pts
#
#
# def random_color_jitter(img, power=10, u=0.5):
#     if np.random.random() < u:
#         noise = np.random.randint(0, power, img.shape[:2])
#         zitter = np.zeros_like(img)
#         zitter[:,:,1] = noise
#         img = cv2.add(img, zitter)
#     return img
#
#
# def random_shift_rotate(image, pts,
#                   shift_limit=(-0.05, 0.05),
#                   rotate_limit=(-10, 10),
#                   borderMode=cv2.BORDER_CONSTANT, u=0.5):
#     if np.random.random() < u:
#         height, width, channel = image.shape
#
#         angle = np.random.uniform(rotate_limit[0], rotate_limit[1])  # degree
#         dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
#         dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)
#
#         cc = np.math.cos(angle / 180 * np.math.pi)
#         ss = np.math.sin(angle / 180 * np.math.pi)
#         rotate_matrix = np.array([[cc, -ss], [ss, cc]])
#
#         box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
#         box1 = box0 - np.array([width / 2, height / 2])
#         box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])
#
#         box0 = box0.astype(np.float32)
#         box1 = box1.astype(np.float32)
#         mat = cv2.getPerspectiveTransform(box0, box1)
#         image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
#                                     borderValue=(
#                                         0, 0,
#                                         0,))
#         pts = (pts - 1 / 2.).reshape(-1, 2).dot(rotate_matrix[::-1]).reshape(-1) + 1 / 2.
#         pts[::2] += dx / 100.
#         pts[1::2] += dy / 100.
#
#     return image, pts


def normalize(img):
    img = img.astype(np.float32)
    img /= 255.
    img -= img.mean()
    img /= img.std()
    return img
    # for c in range(3):
    #     img[:,:,c] = img[:,:,c] - img[:,:,c].mean


def resize_(img, pts=None):
    transformation = transform.SimilarityTransform(scale=float(INPUT_SIZE) / img.shape[0])
    img = transform.warp(img, inverse_map=transformation.inverse, output_shape=(INPUT_SIZE, INPUT_SIZE))
    if pts is None:
        return img, transformation
    for i in range(NUM_POINTS // 2):
        pts[2 * i + 1], pts[2 * i] = transformation([pts[2 * i + 1], pts[2 * i]])[0]
    return img, pts


def get_batch_generator(fnames, points, img_dir, augmentation=True):
    def batch_generator():
        while True:
            combined = list(zip(fnames, points))
            shuffle(combined)
            fnames[:], points[:] = zip(*combined)
            for start in range(0, len(fnames), BATCH_SIZE):
                x_batch = []
                end = min(start + BATCH_SIZE, len(fnames))
                batch_filename = fnames[start: end]
                batch_pts = deepcopy(points[start: end])
                for fname, pts in zip(batch_filename, batch_pts):
                    img = imread(join(img_dir, fname), as_grey=False)
                    if len(img.shape) != 3:
                        img = gray2rgb(img)

                    pts[::2] /= img.shape[1]
                    pts[1::2] /= img.shape[0]
                    img = resize(img, (INPUT_SIZE, INPUT_SIZE), mode="wrap", preserve_range=True)
                    img = normalize(img)

                    # img, pts = resize_(img, pts)
                    # img = normalize(img)
                    # pts[::2] /= 100.
                    # pts[1::2] /= 100.


                    if augmentation:
                        # img = random_color_jitter(img, 5)
                        # img, pts = random_horizontal_flip(img, pts)
                        # img, pts = random_shift_rotate(img, pts)
                        pass
                    x_batch.append(img)
                x_batch = np.array(x_batch, np.float32)
                y_batch = np.array(batch_pts, np.float32)
                yield x_batch, y_batch
    return batch_generator


def get_test_batch_generator(fnames, img_dir):
    def test_batch_generator():
        while True:
            for start in range(0, len(fnames), BATCH_SIZE * 2):
                x_batch = []
                end = min(start + BATCH_SIZE * 2, len(fnames))
                batch_filename = fnames[start: end]
                for fname in batch_filename:
                    img = imread(join(img_dir, fname), as_grey=False)
                    if len(img.shape) != 3:
                        img = gray2rgb(img)

                    img = resize(img, (INPUT_SIZE, INPUT_SIZE), mode="wrap", preserve_range=True)
                    img = normalize(img)

                    x_batch.append(img)
                x_batch = np.array(x_batch, np.float32)
                yield x_batch
    return test_batch_generator


def get_model(cn, cs, dense_num, dense_size, csc):
    model = Sequential()

    model.add(Convolution2D(cs, (3, 3), padding='valid', input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.1))

    for i in range(1, cn):
        cs *= csc
        cs = int(cs)
        model.add(Convolution2D(cs, (3, 3), padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.1))


    # model.add(GlobalMaxPooling2D())
    # print(model.output_shape)

    model.add(Flatten())

    for _ in range(dense_num):
        model.add(Dense(dense_size))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        # model.add(Dropout(dp))
    # model.add(Dense(512))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(Dense(512))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(Dense(512))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))

    model.add(Dense(NUM_POINTS))

    model.compile(optimizer=RMSprop(lr=LEARNING_RATE), # SGD(lr=LEARNING_RATE, momentum=MOMENTUM),
                  loss=mean_squared_error)
    return model


def train_detector(train_gt, train_img_dir, fast_train=False):
    if fast_train:
        return
    tr_fnames, val_fnames = train_test_split(list(train_gt.keys()), test_size=0.2, random_state=17)

    # for fname in val_fnames:
    #     call("cp {} {}".format(
    #         join("/home/ilivans/projects/ysda-computer-vision-part1/data/00_input/train/images", fname),
    #         join("/home/ilivans/projects/ysda-computer-vision-part1/data_val/00_input/test/images", "")
    #     ), shell=True)
    # return

    tr_gt = {fn: train_gt[fn] for fn in tr_fnames}
    val_gt = {fn: train_gt[fn] for fn in val_fnames}

    callbacks = [EarlyStopping(monitor='val_loss',
                               patience=8,
                               verbose=1,
                               min_delta=1e-4),
                 ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.1,
                                   patience=4,
                                   verbose=1,
                                   epsilon=1e-4)]
    if not fast_train:
        callbacks.append(ModelCheckpoint(monitor='val_loss',
                                         filepath='facepoints_model.hdf5',
                                         save_best_only=True))

    for dn in (4,):
        for ds in (512,):
            for cn in (5,):
                for cs in (16,):
                    for csc in (3.,):
                        for dp in (0.0,):
                            model = get_model(cn, cs, dn, ds, csc)
                            model.fit_generator(generator=get_batch_generator(list(tr_gt.keys()), list(tr_gt.values()), train_img_dir,
                                                                          augmentation=True)(),
                                            steps_per_epoch=np.ceil(float(len(tr_gt)) / float(BATCH_SIZE)),
                                            epochs=EPOCHS if not fast_train else 1,
                                            verbose=2,
                                            callbacks=callbacks,
                                            validation_data=get_batch_generator(list(val_gt.keys()), list(val_gt.values()), train_img_dir,
                                                                                augmentation=False)(),
                                            validation_steps=np.ceil(float(len(val_gt)) / float(BATCH_SIZE)))
                            print(dn, ds, cn, cs, csc, end="\n\n\n\n")

def read_img_shapes(fnames, img_dir):
    img_shapes = []
    for fname in fnames:
        img_shapes.append(imread(join(img_dir, fname)).shape[:2])
    return np.asarray(img_shapes)


def detect(model, test_img_dir):
    fnames = listdir(test_img_dir)
    detected_points = model.predict_generator(generator=get_test_batch_generator(fnames, test_img_dir)(),
                                              steps=np.ceil(len(fnames) / float(BATCH_SIZE * 2)))
    imgs_shapes = read_img_shapes(fnames, test_img_dir)
    detected_points[:, ::2] *= imgs_shapes[:, 1:]
    detected_points[:, 1::2] *= imgs_shapes[:, :1]
    return {fname: points for fname, points in zip(fnames, detected_points)}


if __name__=="__main__":
    data_dir = "../../data/00_input"
    train_dir = join(data_dir, 'train')
    train_gt = read_csv(join(train_dir, 'gt.csv'))
    train_img_dir = join(train_dir, 'images')

    train_detector(train_gt, train_img_dir, False)

    test_dir = join(data_dir, 'test')
    test_img_dir = join(test_dir, 'images')
    code_dir = dirname(abspath(__file__))
    # model = load_model(join(code_dir, 'facepoints_model.hdf5'))
    # detected_points = detect(model, test_img_dir)
    # for v in detected_points.values():
    #     print(v)
