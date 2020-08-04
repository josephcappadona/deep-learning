from .imports import *
from .utils import format_history
from collections import defaultdict

__all__ = ['VAE', 'Sampling']

class Sampling(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(Model):

    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

        self.history = defaultdict(list)
    
    def update_history(self, train_history, test_history):
        for metric, value in train_history.items():
            self.history[metric].append(value)
        for metric, value in test_history.items():
            self.history[f'val_{metric}'].append(value)
    
    def fit(self, X_train, X_test=None, y_train=None, y_test=None, datagen=None, n_steps=None, batch_size=32, num_epochs=5, verbose=2):
        datagen = datagen or ImageDataGenerator()
        train_generator = datagen.flow(X_train, y_train if y_train is not None else X_train, batch_size=batch_size)
        
        n_steps = n_steps or int(math.ceil(len(X_train) / batch_size))
        for epoch in range(num_epochs):
            for step in range(n_steps):
                X_batch, y_batch = next(train_generator)
                train_history = format_history(self.train_step(X_batch, y_batch))
                if verbose >= 2:
                    if step % int(math.ceil(n_steps / 5)) == 0:
                        print('.', end='')

            test_history = (format_history(self.test_step(X_test, y_test))
                            if X_test is not None
                            else {})
            self.update_history(train_history, test_history)
            
            if verbose >= 1 and epoch % 1 == 0:
                print(f'Epoch {epoch}:', test_history)
        return self.history

    def get_losses(self, X, y=None):
        z_mean, z_log_var, z = self.encoder(X)
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(X if y is None else y, reconstruction))
        reconstruction_loss *= 28 * 28
        # TODO: https://stats.stackexchange.com/questions/341954/balancing-reconstruction-vs-kl-loss-variational-autoencoder
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5
        total_loss = reconstruction_loss + kl_loss
        return {'loss': total_loss,
                'reconstruction_loss': reconstruction_loss,
                'kl_loss': kl_loss}

    @tf.function
    def train_step(self, X, y=None):
        with tf.GradientTape() as tape:
            losses_dict = self.get_losses(X, y)
        grads = tape.gradient(losses_dict['loss'], self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return losses_dict

    @tf.function
    def test_step(self, X, y=None):
        losses_dict = self.get_losses(X, y)
        return losses_dict
