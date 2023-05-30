from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Input, Conv2D, ReLU, Flatten, Dense, Reshape, Conv2DTranspose, Activation
from keras.layers.normalization.batch_normalization import BatchNormalization
from tensorflow.python.keras import backend
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras.losses import MeanSquaredError
import numpy as np
import os
import pickle


class Autoencoder:

    def __init__(self, input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim):
        self.input_shape = input_shape 
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.latent_space_dim = latent_space_dim

        self.encoder = None
        self.decoder = None
        self.model = None

        self.num_conv_layers = len(conv_filters)
        self.shape_before_bottleneck = None
        self.model_input = None

        self.build()


    def build(self):
        self.build_encoder()
        self.build_decoder()
        self.build_autoencoder()


    #Encoder 

    def build_encoder(self):
        encoder_input = self.add_encoder_input()
        conv_layers = self.add_conv_layers(encoder_input)
        bottleneck = self.add_bottleneck(conv_layers)

        self.model_input = encoder_input

        self.encoder = Model(encoder_input, bottleneck, name="encoder")


    def add_encoder_input(self):
        return Input(shape= self.input_shape, name="encoder_input")
    
    
    def add_conv_layers(self, encoder_input):
        x = encoder_input

        for layer_index in range(self.num_conv_layers):
            x = self.add_conv_layer(layer_index, x)

        return x
    

    def add_conv_layer(self, layer_index, x):
        layer_number = layer_index+1
        conv_layer = Conv2D(filters=self.conv_filters[layer_index],
                            kernel_size=self.conv_kernels[layer_index],
                            strides=self.conv_strides[layer_index],
                            padding="same",
                            name=f"encoder_conv_layer_{layer_number}")
        
        x = conv_layer(x)
        x = ReLU(name=f"encoder_relu_{layer_number}")(x)
        conv_layer = BatchNormalization(name=f"encoder_bn_layer_{layer_number}")(x) 
        
        return conv_layer
    
    
    def add_bottleneck(self, conv_layers):
        self.shape_before_bottleneck = backend.int_shape(conv_layers)[1:]

        flatten_layer = Flatten()(conv_layers)
        dense_layer = Dense(self.latent_space_dim, name="encoder_output")(flatten_layer)

        return dense_layer
    

    #Decoder

    def build_decoder(self):
        decoder_input = self.add_decoder_input()
        dense_layer = self.add_dense_layer(decoder_input)
        reshape_layer = self.add_reshape_layer(dense_layer)
        conv_transpose_layers = self.add_conv_transpose_layers(reshape_layer)
        decoder_output = self.add_decoder_output(conv_transpose_layers)

        self.decoder = Model(decoder_input, decoder_output, name="decoder")


    def add_decoder_input(self):

        return Input(shape = self.latent_space_dim, name="decoder_input")
    

    def add_dense_layer(self, decoder_input):
        size_dense_layer = np.prod(self.shape_before_bottleneck)

        dense_layer = Dense(size_dense_layer, name="decoder_dense")(decoder_input)

        return dense_layer
    
    
    def add_reshape_layer(self, dense_layer):
        reshape_layer = Reshape(self.shape_before_bottleneck)(dense_layer)

        return reshape_layer
    

    def add_conv_transpose_layers(self, x):
        for layer_index in reversed(range(1, self.num_conv_layers)):
            x = self.add_conv_transpose_layer(layer_index, x)

        return x


    def add_conv_transpose_layer(self, layer_index, x):
        layer_number = self.num_conv_layers-layer_index

        conv_transpose_layer = Conv2DTranspose(filters=self.conv_filters[layer_index],
                            kernel_size=self.conv_kernels[layer_index],
                            strides=self.conv_strides[layer_index],
                            padding="same",
                            name=f"decoder_conv_layer_{layer_number}")
        
        x = conv_transpose_layer(x)
        x = ReLU(name=f"decoder_relu_{layer_number}")(x)
        x = BatchNormalization(name=f"decoder_bn_layer_{layer_number}")(x) 

        return x


    def add_decoder_output(self, x):
        conv_transpose_layer = Conv2DTranspose(filters=1,
                            kernel_size=self.conv_kernels[0],
                            strides=self.conv_strides[0],
                            padding="same",
                            name=f"decoder_conv_transpose_layer_{self.num_conv_layers}")
        
        x = conv_transpose_layer(x)
        output = Activation("sigmoid", name="sigmoid_layer")(x)

        return output
    

    #Autoencoder

    def build_autoencoder(self):
        model_input = self.model_input
        model_output = self.decoder(self.encoder(model_input))
        self.model = Model(model_input, model_output, name="autoencode")
        

    def compile(self, learning_rate=0.0001):
        optimizer = adam_v2.Adam(learning_rate=learning_rate)
        loss = MeanSquaredError()
        self.model.compile(optimizer=optimizer, loss=loss)


    def train(self, x_train, batch_size, num_epochs):
        self.model.fit(x_train,
                       x_train,
                       batch_size=batch_size,
                       epochs=num_epochs,
                       shuffle=True)    


    def save(self, save_fold="."):
        self.create_folder_if_doesnt_exist(save_fold)
        self.save_params(save_fold)
        self.save_weights(save_fold)


    def create_folder_if_doesnt_exist(self, save_fold):
        if not os.path.exists(save_fold):
            os.makedirs(save_fold)


    def save_params(self, save_fold):
        parameters = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_space_dim
        ]

        save_path = os.path.join(save_fold, "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

        
    def save_weights(self, save_fold):
        save_path = os.path.join(save_fold, "weights.h5")
        self.model.save_weights(save_path)

    @classmethod
    def load(cls, save_fold="."):
        parameters_path = os.path.join(save_fold, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)

        model = Autoencoder(*parameters)

        weights_path = os.path.join(save_fold, "weights.h5")
        model.load_weights(weights_path)

        return model
    

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)


    def reconstruct(self, images):
        latent_representation = self.encoder.predict(images)
        reconstructed_images = self.decoder.predict(latent_representation)

        return reconstructed_images, latent_representation


    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()


if __name__ == "__main__":
    input_shape = [28, 28, 1]
    conv_filters = [32, 64, 64, 64]
    conv_kernels = [3,3,3,3]
    conv_strides = [1, 2, 2, 1]
    latent_space_dim = 2

    model = Autoencoder(input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim)
    model.summary()
    
