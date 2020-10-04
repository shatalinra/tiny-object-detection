import os, sys, logging, argparse
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import matplotlib.pyplot as plt
import autoencoder

# разбираем входящие параметры
parser = argparse.ArgumentParser(description='Tiny object detection training script')
parser.add_argument('--log', help='Path to a log file.')
script_args = parser.parse_args()

# setup logging before anything else
log_format = '%(asctime)s: <%(levelname)s> %(message)s'
if script_args.log:
    try:
        error_stream = logging.StreamHandler()
        error_stream.setLevel(logging.INFO)
        log_file = logging.FileHandler(script_args.log)
        logging.basicConfig(format=log_format, level=logging.INFO, handlers=[error_stream, log_file])
    except OSError as err:
        print("Error while creating log {}: {}. Exiting...".format(err.filename, err.strerror))
        input("Press Enter to continue...")
        sys.exit(1)
else:
    logging.basicConfig(format=log_format, level=logging.INFO)

# now we can setup hooks for uncaught exceptions
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception

# giving message that log is indeed initialized
print("Log initialized")

#tf.debugging.set_log_device_placement(True)

train_file = tf.io.read_file('train.jpg')
train_image = tf.io.decode_jpeg(train_file)
train_image = tf.expand_dims(train_image, 0)
train_image = tf.cast(train_image, dtype = tf.float32) / 255

test_file = tf.io.read_file('test.jpg')
test_image = tf.io.decode_jpeg(test_file)
test_image = tf.expand_dims(test_image, 0)
test_image = tf.cast(test_image, dtype = tf.float32) / 255

test_image_size = test_image.get_shape().as_list()[1:3]
patch_size = 64
stride = 16

ksizes = [1, patch_size, patch_size, 1] 
strides = [1, stride, stride, 1]
rates = [1, 1, 1, 1]

train_image_patches = tf.image.extract_patches(train_image, ksizes, strides, rates, 'VALID')
train_image_patches = tf.reshape(train_image_patches, [-1, patch_size, patch_size, 3])

test_image_patches = tf.image.extract_patches(test_image, ksizes, strides, rates, 'VALID')
test_patches_count = test_image_patches.get_shape().as_list()[1:3]
test_image_patches = tf.reshape(test_image_patches, [-1, patch_size, patch_size, 3])

network = autoencoder.Autoencoder(patch_size)
try:
    network.load("model")
except IOError:
    network.train(test_image_patches, "model")

reconstructed = network(test_image_patches)

debug = True
if debug:
    
    fig=plt.figure(figsize=(15, 4))
    offset = int(368 / stride) * test_patches_count[1] + int(480 / stride)
    cols = 10
    for i in range(1, cols):
        fig.add_subplot(2, cols, i)
        plt.imshow(test_image_patches[offset + i])
        fig.add_subplot(2, cols, cols + i)
        plt.imshow(reconstructed[offset + i])
    plt.show()

# search for patch with max loss
max_loss = tf.constant(0, dtype=tf.float32)
max_loss_pos = None
for i in range(test_patches_count[0]):
    for j in range(test_patches_count[1]):
        src_patch = test_image_patches[i * test_patches_count[1] + j]
        reconstructed_patch = reconstructed[i * test_patches_count[1] + j]
        diff = tf.math.abs(src_patch - reconstructed_patch)
        loss = tf.reduce_mean(diff)
        if i == int(368 / stride) and j == int(480 / stride):
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
x = max_loss_pos[1] / test_patches_count[1]
y = max_loss_pos[0] / test_patches_count[0]
width = patch_size / test_image_size[0]
height = patch_size / test_image_size[1]
box = tf.convert_to_tensor([[[x, y, x + width, y + height]]], dtype=tf.float32)
colors = tf.convert_to_tensor([[1, 0, 0]], dtype=tf.float32)
output = tf.image.draw_bounding_boxes(test_image, box, colors)

plt.imshow(output[0])
plt.show()


logging.info("max loss is %f", max_loss)

fig=plt.figure(figsize=(15, 4))
fig.add_subplot(1, 2, 1)
plt.imshow(test_image_patches[max_loss_pos[0] * test_patches_count[0] + max_loss_pos[1]])
fig.add_subplot(1, 2, 2)
plt.imshow(reconstructed[max_loss_pos[0] * test_patches_count[0] + max_loss_pos[1]])
plt.show()