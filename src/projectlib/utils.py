from .imports import *

__all__ = ['format_history', 'resize']


def format_history(history):
    return {k: v.numpy() for k, v in history.items()}

def get_ordinal_encoder(labels):
    labels = sorted(set(labels))
    n_unique = len(set(labels))
    encoding = dict(list(zip(labels, list(range(n_unique)))))
    def encoder(label):
        return encoding[label]
    return encoder

def resize(im, desired_size, maintain_aspect_ratio=True, padding_color=(0, 0, 0)):
    # adapted from https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/#using-opencv
    old_size = im.shape[:2] # old_size is in (height, width) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=padding_color)

    return new_im