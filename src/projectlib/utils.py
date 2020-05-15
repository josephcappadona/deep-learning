from .imports import *

__all__ = ['format_history']


def format_history(history):
    return {k: v.numpy() for k, v in history.items()}

def get_ordinal_encoder(labels):
    labels = sorted(set(labels))
    n_unique = len(set(labels))
    encoding = dict(list(zip(labels, list(range(n_unique)))))
    def encoder(label):
        return encoding[label]
    return encoder