import tensorflow as tf
from pathlib import Path

class Autoencoder(object):
    """autoencoder for tiny object detection"""
    def __init__(self, patch_size, *args, **kwargs):
        self._model = None
        self._patch_size = patch_size
        return super().__init__(*args, **kwargs)

    def load(self, path):
         self._model = tf.keras.models.load_model(path + "\\v4")

    def train(self, image_patches, path):
        dir = Path(path)
        dir.mkdir(0o777, True, True)

        # in the end we are going to use 8 layers but for better convergence we are going to train them incrementally
        # in order to save progress while tuning next layers, we are going to cache results. so try load them first
        conv1_layer = None
        conv2_layer = None
        conv3_layer = None
        deconv3_layer = None
        deconv2_layer = None
        deconv1_layer = None
        try:
            model3 = tf.keras.models.load_model(path + "\\v3")
            conv1_layer = model3.get_layer("conv1")
            conv2_layer = model3.get_layer("conv2")
            conv3_layer = model3.get_layer("conv3")
            deconv3_layer = model3.get_layer("deconv3")
            deconv2_layer = model3.get_layer("deconv2")
            deconv1_layer = model3.get_layer("deconv1")
        except IOError:
            try:
                model2 = tf.keras.models.load_model(path + "\\v2")
                conv1_layer = model2.get_layer("conv1")
                conv2_layer = model2.get_layer("conv2")
                deconv2_layer = model2.get_layer("deconv2")
                deconv1_layer = model2.get_layer("deconv1")
            except IOError:
                try:
                    model1 = tf.keras.models.load_model(path + "\\v1")
                    conv1_layer = model1.get_layer("conv1")
                    deconv1_layer = model1.get_layer("deconv1")
                except IOError:
                    pass

        # now if we don't have trained first layers, train them
        if not conv1_layer:
            model1 =  tf.keras.Sequential(name = "autoencoder-v1")
            model1.add(tf.keras.Input(shape = [self._patch_size, self._patch_size, 3]))
            model1.add(tf.keras.layers.Conv2D(filters=15, kernel_size = 3, strides=2, padding="same", activation = 'relu', name = "conv1"))
            model1.add(tf.keras.layers.Conv2DTranspose(filters=3, kernel_size = 3, strides=2, padding="same", activation = 'relu', name = "deconv1"))
            print("Training autoencoder v1")
            self._train_model(model1, image_patches, 0.002, 1600)
            model1.save(path+ '\\v1')

            # extract layers from model
            conv1_layer = model1.get_layer("conv1")
            deconv1_layer = model1.get_layer("deconv1")

        # freeze first layers
        conv1_layer.trainable = False
        deconv1_layer.trainable = False

        # if we don't have trained second layers, also train them
        if not conv2_layer:
            model2 =  tf.keras.Sequential(name = "autoencoder-v2")
            model2.add(tf.keras.Input(shape = [self._patch_size, self._patch_size, 3]))
            model2.add(conv1_layer)
            model2.add(tf.keras.layers.Conv2D(filters=15, kernel_size = 3, strides=2, padding="same", activation = 'relu', name = "conv2"))
            model2.add(tf.keras.layers.Conv2DTranspose(filters=15, kernel_size = 3, strides=2, padding="same", activation = 'relu', name = "deconv2"))
            model2.add(deconv1_layer)
            print("Training autoencoder v2")
            self._train_model(model2, image_patches, 0.002, 1600)
            model2.save(path + "\\v2")

            # extract layers from model
            conv2_layer = model2.get_layer("conv2")
            deconv2_layer = model2.get_layer("deconv2")

        # freeze second layers
        conv2_layer.trainable = False
        deconv2_layer.trainable = False

        # if we don't have trained third layers, also train them
        if not conv3_layer:
            model3 =  tf.keras.Sequential(name = "autoencoder-v3")
            model3.add(tf.keras.Input(shape = [self._patch_size, self._patch_size, 3]))
            model3.add(conv1_layer)
            model3.add(conv2_layer)
            model3.add(tf.keras.layers.Conv2D(filters=30, kernel_size = 3, strides=2, padding="same", activation = 'relu', name = "conv3"))
            model3.add(tf.keras.layers.Conv2DTranspose(filters=15, kernel_size = 3, strides=2, padding="same", activation = 'relu', name = "deconv3"))
            model3.add(deconv2_layer)
            model3.add(deconv1_layer)
            print("Training autoencoder v3")
            self._train_model(model3, image_patches, 0.005, 1000)
            model3.save(path + "\\v3")

            # extract layers from model
            conv3_layer = model3.get_layer("conv3")
            deconv3_layer = model3.get_layer("deconv3")
        
        # freeze third layers
        conv3_layer.trainable = False
        deconv3_layer.trainable = False

        # train last layers
        model4 =  tf.keras.Sequential(name = "autoencoder-v4")
        model4.add(tf.keras.Input(shape = [self._patch_size, self._patch_size, 3]))
        model4.add(conv1_layer)
        model4.add(conv2_layer)
        model4.add(conv3_layer)
        model4.add(tf.keras.layers.Conv2D(filters=18, kernel_size = 3, strides=2, padding="same", activation = 'relu', name = "conv4"))
        model4.add(tf.keras.layers.Conv2DTranspose(filters=30, kernel_size = 3, strides=2, padding="same", activation = 'relu', name = "deconv4"))
        model4.add(deconv3_layer)
        model4.add(deconv2_layer)
        model4.add(deconv1_layer)
        print("Training autoencoder v4")
        self._train_model(model4, image_patches, 0.005, 1000)
        model4.save(path + "\\v4")
            
        self._model = model4

    def _train_model(self, model, image_patches, rate, epochs):
        train_data = tf.data.Dataset.from_tensor_slices(image_patches).batch(512).shuffle(buffer_size = 1024)
        optimizer = tf.optimizers.RMSprop(learning_rate = rate) 
        mse_loss = tf.keras.losses.MeanSquaredError() 
        loss_metric = tf.keras.metrics.Mean()

        model.summary()
        for epoch in range(epochs):
            # Iterate over the batches of the dataset.
            for step, x_batch_train in enumerate(train_data):
                with tf.GradientTape() as tape:
                    reconstructed = model(x_batch_train)
                    # Compute reconstruction loss
                    loss = mse_loss(x_batch_train, reconstructed)

                grads = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

                loss_metric(loss)

            if epoch % 10 == 0:
                print("epoch %d: mean loss = %.6f" % (epoch, loss_metric.result()))

    def __call__(self, data):
        return self._model(data)