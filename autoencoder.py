import tensorflow as tf
from pathlib import Path

class AutoencoderLogger(tf.keras.callbacks.Callback):
    def __init__(self, test_data, test_rate):
        '''
        display: Number of epochs to wait before outputting loss
        '''
        self.test_data = test_data
        self.test_rate = test_rate
        self.loss = tf.losses.MeanSquaredError()
        self.last_loss = None
        
    def on_epoch_end(self, epoch, logs = None):
         if epoch % self.test_rate != 0: return

         reconstructed_data = self.model(self.test_data)
         loss = self.loss(self.test_data, reconstructed_data)

         delta = loss
         if self.last_loss:
             delta = self.last_loss - loss
         self.last_loss = loss

         print('Epoch {0}, loss {1:0.6f}, change {2:0.6f}'.format(epoch, loss, delta))

class Autoencoder(object):
    """autoencoder for tiny object detection"""
    def __init__(self, patch_size, *args, **kwargs):
        self._model = None
        self._patch_size = patch_size
        return super().__init__(*args, **kwargs)

    def load(self, path):
         self._model = tf.keras.models.load_model(path + "\\v1")
         self._model.summary()

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

        # use same min loss change value for training
        min_loss_change = tf.constant(0.000001, dtype=tf.float32)
        init_attempts = 20

        # in choosing architecture several advices were used
        # 1) kernel size divisible by stride https://distill.pub/2016/deconv-checkerboard/
        # 2) using several layers with small kernels lead to better learning than one with large kernel


        # now if we don't have trained first layers, train them
        if not conv1_layer:
            best_model = None
            best_loss = tf.constant(100.0, dtype=tf.float32)
            for init_attempt in range(init_attempts):
                model1 =  tf.keras.Sequential(name = "autoencoder-v1")
                model1.add(tf.keras.Input(shape = [self._patch_size, self._patch_size, 3]))
                model1.add(tf.keras.layers.Conv2D(filters=5, kernel_size = 2, strides = 2, activation = 'relu', name = "conv1"))
                model1.add(tf.keras.layers.Conv2DTranspose(filters=3, kernel_size = 2, strides = 2, activation = 'relu', name = "deconv1"))
                print("Training autoencoder v1: attempt", init_attempt)
                loss = self._train_model(model1, image_patches, min_loss_change, 5000)
                if loss < best_loss:
                    best_loss = loss
                    best_model = model1

            best_model.save(path + '\\v1')
            print("best loss is %.6f" % best_loss)

            self._model = best_model
            return

            # extract layers from model
            conv1_layer = best_model.get_layer("conv1")
            deconv1_layer = best_model.get_layer("deconv1")

        # freeze first layers
        conv1_layer.trainable = False
        deconv1_layer.trainable = False

        # if we don't have trained second layers, also train them
        if not conv2_layer:
            best_model = None
            best_loss = tf.constant(100.0, dtype=tf.float32)
            for init_attempt in range(init_attempts):
                model2 =  tf.keras.Sequential(name = "autoencoder-v2")
                model2.add(tf.keras.Input(shape = [self._patch_size, self._patch_size, 3]))
                model2.add(conv1_layer)
                model2.add(tf.keras.layers.Conv2D(filters=2, kernel_size = 2, strides=2, activation = 'relu', name = "conv2"))
                model2.add(tf.keras.layers.Conv2DTranspose(filters=5, kernel_size = 2, strides=2, activation = 'relu', name = "deconv2"))
                model2.add(deconv1_layer)
                print("Training autoencoder v1: attempt", init_attempt)
                loss = self._train_model(model2, image_patches, min_loss_change, 5000)
                if loss < best_loss:
                    best_loss = loss
                    best_model = model2

            best_model.save(path + '\\v2')
            print("best loss is %.6f" % best_loss)

            self._model = best_model
            return

            # extract layers from model
            conv2_layer = best_model.get_layer("conv2")
            deconv2_layer = best_model.get_layer("deconv2")

      

        # freeze second layers
        conv2_layer.trainable = False
        deconv2_layer.trainable = False

        # if we don't have trained third layers, also train them
        if not conv3_layer:
            best_model = None
            best_loss = tf.constant(100.0, dtype=tf.float32)
            for init_attempt in range(init_attempts):
                model3 =  tf.keras.Sequential(name = "autoencoder-v3")
                model3.add(tf.keras.Input(shape = [self._patch_size, self._patch_size, 3]))
                model3.add(conv1_layer)
                model3.add(conv2_layer)
                model3.add(tf.keras.layers.Conv2D(filters=12, kernel_size = 4, strides = 4, activation = 'relu', name = "conv3"))
                model3.add(tf.keras.layers.Conv2DTranspose(filters=12, kernel_size = 4, strides=4, activation = 'relu', name = "deconv3"))
                model3.add(deconv2_layer)
                model3.add(deconv1_layer)
                print("Training autoencoder v3: attempt", init_attempt)
                loss = self._train_model(model3, image_patches, min_loss_change, 5000)
                if loss < best_loss:
                    best_loss = loss
                    best_model = model3

            best_model.save(path + "\\v3")
            print("best loss is %.6f" % best_loss)

            self._model = best_model
            return

            # extract layers from model
            conv3_layer = best_model.get_layer("conv3")
            deconv3_layer = best_model.get_layer("deconv3")
        
        # freeze third layers
        conv3_layer.trainable = False
        deconv3_layer.trainable = False

        # train last layers
        model4 =  tf.keras.Sequential(name = "autoencoder-v4")
        model4.add(tf.keras.Input(shape = [self._patch_size, self._patch_size, 3]))
        model4.add(conv1_layer)
        model4.add(conv2_layer)
        model4.add(conv3_layer)
        model4.add(tf.keras.layers.Conv2D(filters=15, kernel_size = 3, strides=2, padding="same", activation = 'relu', name = "conv4"))
        model4.add(tf.keras.layers.Conv2DTranspose(filters=9, kernel_size = 3, strides=3, padding="same", activation = 'relu', name = "deconv4"))
        model4.add(deconv3_layer)
        model4.add(deconv2_layer)
        model4.add(deconv1_layer)
        print("Training autoencoder v4")
        self._train_model(model4, image_patches, min_loss_change)
        model4.save(path + "\\v4")
            
        self._model = model4

    def _train_model(self, model, image_patches, min_loss_change, max_epoch):
        model.summary()
        model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.RMSprop())

        logging = AutoencoderLogger(image_patches, 50)
        stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100, mode='min')
        history = model.fit(image_patches, image_patches, 512, max_epoch, verbose=0, callbacks=[logging, stopping])
        return history.history["loss"][-1]

    def __call__(self, data):
        return self._model(data)