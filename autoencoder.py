import tensorflow as tf

class Autoencoder(object):
    """autoencoder for tiny object detection"""
    def __init__(self, patch_size, *args, **kwargs):
        self._model = None
        self._patch_size = patch_size
        return super().__init__(*args, **kwargs)

    def load(self, path):
         self._model = tf.keras.models.load_model(path)

    def train(self, image_patches, path):
        # first we are going to built autoencoder with just two layers
        # try to load first if it is already cached
        model1 = None
        try:
            model1 = tf.keras.models.load_model("v1")
        except IOError:
            # train if it does not exist
            model1 =  tf.keras.Sequential(name = "autoencoder-v1")
            model1.add(tf.keras.Input(shape = [self._patch_size, self._patch_size, 3]))
            model1.add(tf.keras.layers.Conv2D(filters=48, kernel_size = 3, strides=2, padding="same", activation = 'relu', name = "conv1"))
            model1.add(tf.keras.layers.Conv2DTranspose(filters=3, kernel_size = 3, strides=2, padding="same", activation = 'relu', name = "deconv1"))
            print("Training autoencoder v1")
            self._train_model(model1, image_patches)
            model1.save('v1')

        # extract trained convolutional layers, we are going to use them for next models
        conv1_layer = model1.get_layer("conv1")
        deconv1_layer = model1.get_layer("deconv1")

        # freeze them
        conv1_layer.trainable = False
        deconv1_layer.trainable = False

        # now we go for second layer based on previous results
        model2 =  tf.keras.Sequential(name = "autoencoder-v2")
        model2.add(tf.keras.Input(shape = [self._patch_size, self._patch_size, 3]))
        model2.add(conv1_layer)
        model2.add(tf.keras.layers.Conv2D(filters=48, kernel_size = 3, strides=2, padding="same", activation = 'relu', name = "conv2"))
        model2.add(tf.keras.layers.Conv2DTranspose(filters=48, kernel_size = 3, strides=2, padding="same", activation = 'relu', name = "deconv2"))
        model2.add(deconv1_layer)
        print("Training autoencoder v2")
        self._train_model(model2, image_patches)

        model2.save(path)
        self._model = model2

    def _train_model(self, model, image_patches):
        epochs = 800
        train_data = tf.data.Dataset.from_tensor_slices(image_patches).batch(512).shuffle(buffer_size = 1024)
        optimizer = tf.optimizers.RMSprop(learning_rate = 0.002) 
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