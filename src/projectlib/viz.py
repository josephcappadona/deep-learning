from .imports import *

__all__ = ['view_images', 'plot']


def view_images(ims, decoder, encoder):
    # TODO: change inputs to not require encoder+decoder
    input_shape = ims[0].shape
    decoded_ims = decoder(encoder(ims)) # TODO: , training=False ?

    plt.figure(figsize=(20, 4))
    n = len(ims)
    for i in range(n):
        # display original
        im = ims[i].reshape(*input_shape[:-1]) if input_shape[-1] == 1 else ims[i]
        ax = plt.subplot(2, n, i+1)
        plt.imshow(im)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        decoded_im = decoded_ims[i].numpy()
        decoded_im = decoded_ims[i].numpy().reshape(*input_shape[:-1]) if input_shape[-1] == 1 else decoded_ims[i].numpy()
        ax = plt.subplot(2, n, i + n + 1)
        plt.imshow(decoded_im)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

def plot(labels, plots, title='', xlabel=None, ylabel=None, legend_loc='best'):
    for p in plots:
        plt.plot(p)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(labels, loc=legend_loc)
    plt.show()
