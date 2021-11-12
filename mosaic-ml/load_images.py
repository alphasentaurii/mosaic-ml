import os
import numpy as np
from keras.preprocessing import image
from tqdm import tqdm
import time
import datetime as dt
from ensemble import proc_time

def read_channels(channels, w, h, d, exp=None, color_mode='rgb'):
    """Loads PNG image data and converts to 3D arrays.
    **args
    channels: tuple of image frames (original, source, gaia)
    w: width
    h: height
    d: depth
    **kwargs
    exp: "expand" dimensions: (exp, w, h, 3). Set to 3 for predictions, None for training (default)

    """
    t = (w, h)
    image_frames = [image.load_img(c, color_mode=color_mode, target_size=t) for c in channels]
    img = np.array([image.img_to_array(i) for i in image_frames])
    if exp == None:
        img = img.reshape(w, h, d)
    else:
        img = img.reshape(exp, w, h, 3)
    return img


def get_labeled_image_paths(i, img_path):
    neg = (
        f"{img_path}/0/{i}/{i}.png",
        f"{img_path}/0/{i}/{i}_source.png",
        f"{img_path}/0/{i}/{i}_gaia.png"
        )
    pos = (
        f"{img_path}/1/{i}/{i}.png",
        f"{img_path}/1/{i}/{i}_source.png",
        f"{img_path}/1/{i}/{i}_gaia.png"
        )
    return neg, pos


def detector_training_images(data, img_path, w, h, d, exp):
    idx = list(data.index)
    files, labels = [], []
    for i in idx:
        neg, pos = get_labeled_image_paths(i, img_path)
        if os.path.exists(neg[0]):
            files.append(neg)
            labels.append(0)
        elif os.path.exists(pos[0]):
            files.append(pos)
            labels.append(1)
        else:
            # print(f"missing: {i}")
            idx.remove(i)
    img = []
    for ch1, ch2, ch3 in tqdm(files):
        img.append(read_channels([ch1, ch2, ch3], w, h, d, exp=exp))
    X, y = np.array(img, np.float32), np.array(labels)
    return (idx, X, y)


def detector_prediction_images(X, img_path, w, h, d, exp):
    image_files = []
    idx = list(X.index)
    for i in idx:
        img_frames = (
            f"{img_path}/{i}/{i}.png",
            f"{img_path}/{i}/{i}_source.png",
            f"{img_path}/{i}/{i}_gaia.png"
            )
        if os.path.exists(img_frames[0]):
            image_files.append(img_frames)
        else:
            idx.remove(i)
    t_start = time.time()
    start = dt.datetime.fromtimestamp(t_start).strftime("%m/%d/%Y - %I:%M:%S %p")
    print(f"\n[i] LOADING IMAGES  ***{start}***")
    img = []
    for ch1, ch2, ch3 in tqdm(image_files):
        img.append(read_channels([ch1, ch2, ch3], w, h, d, exp=exp))
    images = np.array(img, np.float32)
    t_end = time.time()
    end = dt.datetime.fromtimestamp(t_end).strftime("%m/%d/%Y - %I:%M:%S %p")
    print(f"\n[i] IMAGES LOADED ***{end}***")
    proc_time(t_start, t_end)
    return idx, images
