# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # opencv video example

import cv2
from PIL import Image


cap = cv2.VideoCapture('armada_vs_hbox.mp4')

n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(n_frames)

cap.set(cv2.CAP_PROP_POS_FRAMES, 99)
res, frame = cap.read()
Image.fromarray(frame[...,::-1], 'RGB').resize((256, 256)).show()


def resize(im, desired_size, maintain_aspect_ratio=True, padding_color=(0, 0, 0)):
    # adapted from https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/#using-opencv
    old_size = im.shape[:2] # old_size is in (height, width) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=padding_color)

    return new_im


print(frame.shape)
print(resize(frame, 256).shape)

# +
import random
import numpy as np

def sample_frames_from_video(video_path, k, technique='random'):
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    k = min(k, n_frames)
    
    if technique == 'first':
        idxs = list(range(k))
    elif technique == 'random':
        idxs = random.sample(range(n_frames), k)
    
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        _, frame = cap.read()
        frames.append(resize(frame, 256))
        
    return idxs, np.array(frames)


# -

idxs, frames = sample_frames_from_video('armada_vs_hbox.mp4', 128)
print(idxs)
print(frames.shape)

# +
width, height, channels = frames.shape[1:]

from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(frames, test_size=0.1, random_state=42)
# -

X_train.shape

X_test.shape

# +
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(featurewise_center=True,
                             featurewise_std_normalization=True,
                             zoom_range=0.1)
datagen.fit(X_train)
train_iterator = datagen.flow(X_train, X_train, batch_size=32)
test_iterator = datagen.flow(X_test, X_test, batch_size=64)
print('Batches train=%d, test=%d' % (len(train_iterator), len(test_iterator)))
# -

from . import autoencoding_cnns


