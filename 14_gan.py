import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = (x_train.astype("float32") - 127.5) / 127.5
x_train = np.expand_dims(x_train, axis=-1)
generator = tf.keras.Sequential([layers.Input(shape=(100,)), layers.Dense(7*7*128), layers.Reshape((7,7,128)), layers.BatchNormalization(), layers.ReLU(), layers.Conv2DTranspose(64,4,strides=2,padding="same"), layers.BatchNormalization(), layers.ReLU(), layers.Conv2DTranspose(1,4,strides=2,padding="same",activation="tanh")])
discriminator = tf.keras.Sequential([layers.Input(shape=(28,28,1)), layers.Conv2D(64,4,strides=2,padding="same"), layers.LeakyReLU(0.2), layers.Conv2D(128,4,strides=2,padding="same"), layers.LeakyReLU(0.2), layers.Flatten(), layers.Dense(1)])
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
g_opt = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
d_opt = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
batch_size, epochs, noise_dim = 128, 20, 100
dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(60000).batch(batch_size, drop_remainder=True)
@tf.function
def train_step(real_images):
    noise = tf.random.normal([batch_size, noise_dim])
    with tf.GradientTape() as gt, tf.GradientTape() as dt:
        fake_images = generator(noise, training=True)
        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(fake_images, training=True)
        g_loss = loss_fn(tf.ones_like(fake_output), fake_output)
        d_loss = loss_fn(tf.ones_like(real_output), real_output) + loss_fn(tf.zeros_like(fake_output), fake_output)
    g_opt.apply_gradients(zip(gt.gradient(g_loss, generator.trainable_variables), generator.trainable_variables))
    d_opt.apply_gradients(zip(dt.gradient(d_loss, discriminator.trainable_variables), discriminator.trainable_variables))
    return g_loss, d_loss
for epoch in range(epochs):
    for images in dataset: g_loss, d_loss = train_step(images)
    print("Epoch:", epoch+1, "Generator Loss:", float(g_loss), "Discriminator Loss:", float(d_loss))
images = generator(tf.random.normal([16, noise_dim]), training=False)
plt.figure(figsize=(8,8))
for i in range(16):
    plt.subplot(4,4,i+1); plt.imshow(images[i,:,:,0]*0.5+0.5, cmap="gray"); plt.axis("off")
plt.tight_layout(); plt.show()
