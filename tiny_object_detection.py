import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import matplotlib.pyplot as plt

raw_file = tf.io.read_file('bolt-sample.jpg')
image = tf.io.decode_jpeg(raw_file)
image = tf.expand_dims(image, 0)
image_float = tf.cast(image, dtype = tf.float32) / 255

image_size = image.get_shape().as_list()[1:3]
patch_size = 64
stride = 16

ksizes = [1, patch_size, patch_size, 1] 
strides = [1, stride, stride, 1]
rates = [1, 1, 1, 1]


image_patches = tf.image.extract_patches(image_float, ksizes, strides, rates, 'VALID')
patches_count = image_patches.get_shape().as_list()[1:3]
image_patches = tf.reshape(image_patches, [-1, patch_size, patch_size, 3])

model = None
try:
    model = tf.keras.models.load_model("model")
except IOError:
    model = None

if model == None:
    model = tf.keras.Sequential(name = "autoencoder")
    model.add(tf.keras.Input(shape = [patch_size, patch_size, 3]))
    model.add(tf.keras.layers.Conv2D(filters=48, kernel_size = 3, strides=2, padding="same", activation = 'relu', name = "conv1"))
    #model.add(tf.keras.layers.Conv2D(filters=1, kernel_size = 3, strides=2, padding="same", activation = 'relu', name = "conv2"))
    #model.add(tf.keras.layers.Conv2D(filters=3, kernel_size = 3, strides=2, padding="same", activation = 'relu', name = "conv3"))
    #model.add(tf.keras.layers.Flatten())
    #model.add(tf.keras.layers.Dense(10))
    #model.add(tf.keras.layers.Dense(units=8*8*80, activation='relu'))
    #model.add(tf.keras.layers.Reshape(target_shape=(8, 8, 80)))
    #model.add(tf.keras.layers.Conv2DTranspose(filters=1, kernel_size = 3, strides=2, padding="same", activation = 'relu', name = "deconv3"))
    #model.add(tf.keras.layers.Conv2DTranspose(filters=48, kernel_size = 3, strides=2, padding="same", activation = 'relu', name = "deconv2"))
    model.add(tf.keras.layers.Conv2DTranspose(filters=3, kernel_size = 3, strides=2, padding="same", activation = 'relu', name = "deconv1"))
    model.summary()

    epochs = 500
    train_data = tf.data.Dataset.from_tensor_slices(image_patches).batch(128).shuffle(buffer_size = 1024)
    optimizer = tf.optimizers.RMSprop() 
    mse_loss = tf.keras.losses.MeanSquaredError() 
    loss_metric = tf.keras.metrics.Mean()

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

debug = True
if debug:
    
    fig=plt.figure(figsize=(15, 4))
    offset = 32 * patches_count[0] + 9
    cols = 10
    for i in range(1, cols):
        fig.add_subplot(2, cols, i)
        plt.imshow(image_patches[offset + i])
        fig.add_subplot(2, cols, cols + i)
        plt.imshow(reconstructed[offset + i])
    plt.show()

# search for patch with max loss
max_loss = tf.constant(0, dtype=tf.float32)
max_loss_pos = None
for i in range(patches_count[1]):
    for j in range(patches_count[0]):
        src_patch = image_patches[i * patches_count[0] + j]
        reconstructed_patch = reconstructed[i * patches_count[0] + j]
        diff = tf.math.abs(src_patch - reconstructed_patch)
        loss = tf.reduce_mean(diff)
        if i == 32 and j == 10:
            fig=plt.figure(figsize=(15, 4))
            fig.add_subplot(1, 2, 1)
            plt.imshow(src_patch)
            fig.add_subplot(1, 2, 2)
            plt.imshow(reconstructed_patch)
            plt.show()

        if loss > max_loss:
            max_loss = loss
            max_loss_pos = [i, j]

# draw rect on src image
x = max_loss_pos[1] / patches_count[0]
y = max_loss_pos[0] / patches_count[1]
width = patch_size / image_size[0]
height = patch_size / image_size[1]
box = tf.convert_to_tensor([[[x, y, x + width, y + height]]], dtype=tf.float32)
colors = tf.convert_to_tensor([[1, 0, 0]], dtype=tf.float32)
output = tf.image.draw_bounding_boxes(image_float, box, colors)

plt.imshow(output[0])
plt.show()


print("max loss is", max_loss)

fig=plt.figure(figsize=(15, 4))
fig.add_subplot(1, 2, 1)
plt.imshow(image_patches[max_loss_pos[0] * patches_count[0] + max_loss_pos[1]])
fig.add_subplot(1, 2, 2)
plt.imshow(reconstructed[max_loss_pos[0] * patches_count[0] + max_loss_pos[1]])
plt.show()