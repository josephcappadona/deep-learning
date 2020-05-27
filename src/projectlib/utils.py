from .imports import *

__all__ = [
    'format_history',
    'resize',
    'get_random_bbox',
    'crop',
    'sample_frames_from_video'
]


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

def get_random_bbox(im):
    shape = im.shape
    x_min, x_max = sorted([random.randrange(0, im.shape[0]+1) for _ in range(2)])
    y_min, y_max = sorted([random.randrange(0, im.shape[1]+1) for _ in range(2)])
    return (x_min, x_max), (y_min, y_max)

def crop(im, x_min, x_max, y_min, y_max):
    im = im[:]
    return im[y_min:y_max, x_min:x_max]

def sample_frames_from_video(video, k, technique='random', preprocess=None):
    if isinstance(video, str):
        cap = cv2.VideoCapture(video)
    elif isinstance(video, cv2.VideoCapture):
        cap = video
    else:
        raise TypeError(f'Illegal type f{video.__class__.__name__} for argument `video`, must be str or cv2.VideoCapture.')
    
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
        frames.append(frame
                      if preprocess is None
                      else preprocess(frame))
        
    return idxs, np.array(frames)