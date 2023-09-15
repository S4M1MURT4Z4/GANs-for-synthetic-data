import tensorflow as tf
import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Reshape, LeakyReLU, Dropout, UpSampling1D

def build_generator():
    model = Sequential()

    model.add(Dense(9*128, input_dim=128))
    model.add(LeakyReLU(0.2))
    model.add(Reshape((9, 128)))

    model.add(UpSampling1D())
    model.add(Conv1D(128, 5, padding='same'))
    model.add(LeakyReLU(0.2))

    model.add(UpSampling1D())
    model.add(Conv1D(128, 5, padding='same'))
    model.add(LeakyReLU(0.2))

    model.add(Conv1D(128, 4, padding='same'))
    model.add(LeakyReLU(0.2))
    
    model.add(Conv1D(128, 4, padding='same'))
    model.add(LeakyReLU(0.2))

    model.add(Conv1D(1, 4, padding='same', activation='sigmoid'))

    return model

def build_discriminator():
    model = Sequential()

    model.add(Conv1D(32, 5, input_shape=(36, 1)))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    model.add(Conv1D(64,5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    model.add(Conv1D(128,5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    model.add(Conv1D(265,5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))

    return model 



from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

from tensorflow.keras.models import Model

class GAN(Model):

    def __init__(self, gen, dis, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gen = gen
        self.dis = dis

    def train_step(self, batch):
        real_data = batch
        fake_data = self.gen(tf.random.normal((128,128,1)), training=False)

        with tf.GradientTape() as d_tape:
            yhat_real = self.dis(real_data, training=True)
            yhat_fake = self.dis(fake_data, training=True)
            y_hats = tf.concat([yhat_real, yhat_fake], axis=0)

            y_true = tf.concat([tf.zeros_like(yhat_real), tf.ones_like(yhat_fake)], axis=0)

            noise_real = 0.15*tf.random.uniform(tf.shape(yhat_real))
            noise_fake = -0.15*tf.random.uniform(tf.shape(yhat_fake))
            y_true += tf.concat([noise_real, noise_fake], axis=0)

            total_d_loss = self.d_loss(y_true, y_hats)

        dgrad = d_tape.gradient(total_d_loss, self.dis.trainable_variables)
        self.d_opt.apply_gradients(zip(dgrad, self.dis.trainable_variables))

        with tf.GradientTape() as g_tape:
            gen_rows = self.gen(tf.random.normal((128,128,1)), training=True)
            
            predicted_labels = self.dis(gen_rows, training=False)

            total_g_loss = self.g_loss(tf.zeros_like(predicted_labels), predicted_labels)

        ggrad = g_tape.gradient(total_g_loss, self.gen.trainable_variables)
        self.g_opt.apply_gradients(zip(ggrad, self.gen.trainable_variables))

        return {'d_loss': total_d_loss, 'g_loss': total_g_loss}
            

    def compile(self, g_opt, d_opt, g_loss, d_loss, *args, **kwargs):
        super().compile(*args, **kwargs)

        self.g_opt = g_opt
        self.d_opt = d_opt
        self.d_loss = d_loss
        self.g_loss = g_loss


if __name__=='__main__':
    sample_data = pd.read_csv('./sample_submission.csv')
    maxes = sample_data.max()[1:]
            
    g_opt = Adam(learning_rate=0.0001)
    d_opt = Adam(learning_rate=0.00001)
    g_loss = BinaryCrossentropy()
    d_loss = BinaryCrossentropy()


    g = build_generator()
    #o = g.predict(np.random.randn(4, 128))

    d = build_discriminator()

    gan = GAN(g, d)

    gan.compile(g_opt, d_opt, g_loss, d_loss)

    from tensorflow.keras.callbacks import Callback

    class ModelMonitor(Callback):
        def __init__(self, num_rows=4, latent_dim=128):
            self.num_rows = num_rows
            self.latent_dim = latent_dim

        def on_epoch_end(self, epoch, logs=None):
            random_vectors = tf.random.uniform((self.num_rows, self.latent_dim))
            generated_rows = self.model.generator(random_vectors)
            generated_rows *= 255
            generated_rows.numpy()

    s_data = sample_data.loc[:,sample_data.columns!='id']
    hist = gan.fit(s_data, epochs=20)


