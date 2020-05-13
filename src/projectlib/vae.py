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
    
    def fit(self, X_train, X_test, batch_size=32, num_epochs=5, verbose=2):
        train_datagen = ImageDataGenerator()
        train_generator = train_datagen.flow(X_train, batch_size=batch_size)
        
        n_steps = len(X_train) // batch_size
        for epoch in range(num_epochs):
            for step in range(n_steps):
                X_batch = next(train_generator)
                train_history = format_history(self.train_step(X_batch[None, :]))
                if verbose >= 2:
                    if step % (n_steps // 5) == 0:
                        print('.', end='')

            test_history = format_history(self.test_step(X_test[None, :]))
            if verbose >= 1 and epoch % 1 == 0:
                print(f'Epoch {epoch}:', test_history)
            self.update_history(train_history, test_history)
        return self.history

    def get_losses(self, X):
        z_mean, z_log_var, z = self.encoder(X)
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(X, reconstruction))
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
    def train_step(self, X):
        X = X[0]
        with tf.GradientTape() as tape:
            losses_dict = self.get_losses(X)
        grads = tape.gradient(losses_dict['loss'], self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return losses_dict

    @tf.function
    def test_step(self, X):
        X = X[0]
        losses_dict = self.get_losses(X)
        return losses_dict
