from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Input, Conv2D, ReLU, Flatten, Dense, Reshape, Conv2DTranspose, Activation
from keras.layers.normalization.batch_normalization import BatchNormalization
from tensorflow.python.keras import backend
import numpy as np


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
        #self.build_autoencoder()


    def build_encoder(self):
        encoder_input = self.add_encoder_input()
        conv_layers = self.add_conv_layers(encoder_input)
        bottleneck = self.add_bottleneck(conv_layers)

        self.model_input = encoder_input

        self.encoder = Model(encoder_input, bottleneck, name="encoder")

    def build_decoder(self):
        decoder_input = self.add_decoder_input()
        dense_layer = self.add_dense_layer(decoder_input)
        reshape_layer = self.add_reshape_layer(dense_layer)
        conv_transpose_layers = self.add_conv_transpose_layers(reshape_layer)
        decoder_output = self.add_decoder_output(conv_transpose_layers)

        self.decoder = Model(decoder_input, decoder_output, name="decoder")


    def build_autoendoder(self):
        pass


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
                            name=f"encoder_conv_layer_{layer_number}")
        
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
         

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        #self.autoencoder.summary()


if __name__ == "__main__":
    input_shape = [28, 28, 1]
    conv_filters = [32, 64, 64, 64]
    conv_kernels = [3,3,3,3]
    conv_strides = [1, 2, 2, 1]
    latent_space_dim = 2

    model = Autoencoder(input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim)
    model.summary()
    
