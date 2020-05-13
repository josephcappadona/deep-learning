from .imports import *

__all__ = ['format_history']


def format_history(history):
    return {k: v.numpy() for k, v in history.items()}