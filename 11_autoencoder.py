import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

encoder = models.Sequential([layers.Input(shape=(784,)), layers.Dense(128, activation="relu"), layers.Dense(32, activation="relu"), layers.Dense(10, activation="relu")])
decoder = models.Sequential([layers.Input(shape=(10,)), layers.Dense(32, activation="relu"), layers.Dense(128, activation="relu"), layers.Dense(784, activation="sigmoid")])
autoencoder = models.Sequential([encoder, decoder])
autoencoder.summary()
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
history = autoencoder.fit(x_train, x_train, epochs=10, batch_size=256, shuffle=True, validation_data=(x_test, x_test))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Autoencoder Training Loss"); plt.legend(); plt.show()
reconstructed = autoencoder.predict(x_test)
difference = np.abs(x_test - reconstructed)
accuracy = (np.sum(difference < 0.1) / x_test.size) * 100
print("Reconstruction Accuracy: {:.2f}%".format(accuracy))
plt.figure(figsize=(20, 4))
for i in range(10):
    plt.subplot(2, 10, i + 1); plt.imshow(x_test[i].reshape(28, 28), cmap="gray"); plt.axis("off")
    plt.subplot(2, 10, i + 11); plt.imshow(reconstructed[i].reshape(28, 28), cmap="gray"); plt.axis("off")
plt.show()
print("Mean Absolute Reconstruction Error:", np.mean(np.abs(x_test - reconstructed)))
print("Mean Squared Reconstruction Error:", np.mean((x_test - reconstructed) ** 2))
