import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers

(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = (x_train.astype("float32") - 127.5) / 127.5
x_train = np.expand_dims(x_train, axis=-1)
BATCH_SIZE = 128
dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(60000).batch(BATCH_SIZE, drop_remainder=True)

def build_generator():
    return tf.keras.Sequential([layers.Input(shape=(100,)), layers.Dense(7*7*256, use_bias=False), layers.BatchNormalization(), layers.ReLU(), layers.Reshape((7,7,256)), layers.Conv2DTranspose(128, 5, strides=1, padding="same", use_bias=False), layers.BatchNormalization(), layers.ReLU(), layers.Conv2DTranspose(64, 5, strides=2, padding="same", use_bias=False), layers.BatchNormalization(), layers.ReLU(), layers.Conv2DTranspose(1, 5, strides=2, padding="same", activation="tanh")])

def build_discriminator():
    return tf.keras.Sequential([layers.Input(shape=(28,28,1)), layers.Conv2D(64,5,strides=2,padding="same"), layers.LeakyReLU(), layers.Dropout(0.3), layers.Conv2D(128,5,strides=2,padding="same"), layers.LeakyReLU(), layers.Dropout(0.3), layers.Flatten(), layers.Dense(1)])

generator, discriminator = build_generator(), build_discriminator()
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
g_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
d_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)

def generator_loss(fake_output): return cross_entropy(tf.ones_like(fake_output), fake_output)
def discriminator_loss(real_output, fake_output): return cross_entropy(tf.ones_like(real_output), real_output) + cross_entropy(tf.zeros_like(fake_output), fake_output)

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, 100])
    with tf.GradientTape() as gt, tf.GradientTape() as dt:
        fake_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(fake_images, training=True)
        g_loss = generator_loss(fake_output)
        d_loss = discriminator_loss(real_output, fake_output)
    g_optimizer.apply_gradients(zip(gt.gradient(g_loss, generator.trainable_variables), generator.trainable_variables))
    d_optimizer.apply_gradients(zip(dt.gradient(d_loss, discriminator.trainable_variables), discriminator.trainable_variables))
    return g_loss, d_loss

EPOCHS = 20
for epoch in range(EPOCHS):
    for batch in dataset: g_loss, d_loss = train_step(batch)
    print("Epoch:", epoch+1, "Generator Loss:", float(g_loss), "Discriminator Loss:", float(d_loss))
    if epoch == 0 or (epoch+1) % 5 == 0:
        images = generator(tf.random.normal([16,100]), training=False)
        plt.figure(figsize=(8,8))
        for i in range(16):
            plt.subplot(4,4,i+1); plt.imshow((images[i,:,:,0]+1)/2, cmap="gray"); plt.axis("off")
        plt.suptitle("Generated Images - Epoch " + str(epoch+1)); plt.show()

images = generator(tf.random.normal([16,100]), training=False)
plt.figure(figsize=(8,8))
for i in range(16):
    plt.subplot(4,4,i+1); plt.imshow((images[i,:,:,0]+1)/2, cmap="gray"); plt.axis("off")
plt.suptitle("Final Generated Images"); plt.show()
