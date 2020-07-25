import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import matplotlib.pyplot as plt

raw_file = tf.io.read_file('bolt-sample.jpg')
image = tf.io.decode_jpeg(raw_file)
image = tf.cast(image, dtype = tf.float32)

patch_size = 64
stride = 16

ksizes = [1, patch_size, patch_size, 1] 
strides = [1, stride, stride, 1]
rates = [1, 1, 1, 1]
padding='VALID' # or 'SAME'

image = tf.expand_dims(image, 0)
image_patches = tf.image.extract_patches(image, ksizes, strides, rates, padding)
image_patches = tf.reshape(image_patches, [-1, patch_size, patch_size, 3])

train_data = tf.data.Dataset.from_tensor_slices(image_patches).batch(128).shuffle(buffer_size = 1024)
  
optimizer = tf.optimizers.Adam(learning_rate = 0.01) 
mse_loss = tf.keras.losses.MeanSquaredError() 
loss_metric = tf.keras.metrics.Mean()


model = None
try:
    model = tf.keras.models.load_model("model")
except IOError:
    model = None

if model == None:
    model = tf.keras.Sequential(name = "autoencoder")
    model.add(tf.keras.Input(shape = [patch_size, patch_size, 3]))
    model.add(tf.keras.layers.Conv2D(1, 3, 1, "valid", activation = 'relu', name = "conv1"))
    model.add(tf.keras.layers.Conv2DTranspose(3, 3, 1, "valid", activation = 'relu', name = "deconv1"))
    model.summary()

    epochs = 500

    # Iterate over epochs.
    for epoch in range(epochs):
        print("Start of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        for step, x_batch_train in enumerate(train_data):
            with tf.GradientTape() as tape:
                reconstructed = model(x_batch_train)
                # Compute reconstruction loss
                loss = mse_loss(x_batch_train, reconstructed)

            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            loss_metric(loss)

            if step % 100 == 0:
                print("step %d: mean loss = %.4f" % (step, loss_metric.result()))

    model.save('model')


reconstructed = model(image_patches)

fig=plt.figure(figsize=(15, 4))
for i in range(1, 10):
    fig.add_subplot(2, 10, i)
    plt.imshow(image_patches[i] / 255)
    fig.add_subplot(2, 10, 10 + i)
    plt.imshow(reconstructed[i] / 255)
plt.show()