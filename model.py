import theano
import theano.tensor as T

from lasagne import layers, init


class SumLayer(layers.Layer):
    def __init__(self, incoming, axis=-1, **kwargs):
        super(SumLayer, self).__init__(incoming, **kwargs)
        self.axis = axis

    def get_output_for(self, input, **kwargs):
        return input.sum(axis=self.axis)

    def get_output_shape_for(self, input_shape):
        shape = (list(input_shape[0:self.axis]) +
                 list(input_shape[self.axis + 1:]))
        shape = tuple(shape)
        return shape


class RealEmbeddingLayer(layers.Layer):

    def __init__(self, incoming, input_size, output_size,
                 W=init.Normal(), **kwargs):
        super(RealEmbeddingLayer, self).__init__(incoming, **kwargs)
        self.input_size = input_size
        self.output_size = output_size

        self.W = self.add_param(W, (input_size, output_size), name="W")

    def get_output_shape_for(self, input_shape):
        shape = list(input_shape[0:-1]) + [self.output_size]
        shape = tuple(shape)
        return shape

    def get_output_for(self, input, **kwargs):
        return T.tensordot(input, self.W, axes=([input.ndim - 1], [0]))


class RecurrentAccumulationLayer(layers.Layer):

    def __init__(self, incoming, decoder_layer, n_steps, name=None):
        super(RecurrentAccumulationLayer, self).__init__(incoming,
                                                         name=name)
        self.decoder_layer = decoder_layer
        self.n_steps = n_steps

    def get_output_for(self, input, **kwargs):
        result, updates = recurrent_accumulation(input,
                                                 self.decoder_layer,
                                                 self.n_steps)
        return result

    def get_output_shape_for(self, input_shape):
        shape_one_step = self.decoder_layer.get_output_shape_for(input_shape)
        shape = [shape_one_step[0], self.n_steps] + list(shape_one_step[1:])
        shape = tuple(shape)
        return shape


def recurrent_accumulation(X, decomposition_layer, n_steps, **scan_kwargs):

    def step_function(input_i, *args):
        delta_input = layers.get_output(decomposition_layer, input_i)
        return input_i - delta_input

    sequences = []
    outputs_info = [T.zeros(X.shape)]
    non_sequences = layers.get_all_params(decomposition_layer)

    result, updates = theano.scan(fn=step_function,
                                  sequences=sequences,
                                  outputs_info=outputs_info,
                                  non_sequences=non_sequences,
                                  strict=True,
                                  n_steps=n_steps,
                                  **scan_kwargs)
    return result, updates


def build_model(batch_size=None, nb_items=5,
                size_items=8, size_embedding=20,
                size_latent=100):

    # Input to Latent
    l_input = layers.InputLayer((batch_size, nb_items, size_items),
                                name="input")
    l_Z = RealEmbeddingLayer(l_input, size_items, size_embedding,
                             name="emb")
    l_Z_sum = SumLayer(l_Z, axis=1, name="emb_sum")
    l_latent_reconstruction = layers.DenseLayer(l_Z_sum, num_units=size_latent,
                                                name="latent_rec")
    layers_input_to_latent = [
        l_input,
        l_Z,
        l_Z_sum,
        l_latent_reconstruction
    ]

    # Latent to Input
    l_latent = layers.InputLayer((batch_size, size_latent),
                                 name="latent")

    l_Z_sum = layers.DenseLayer(l_latent, num_units=size_embedding)
    decomposition_layer_input = layers.InputLayer((batch_size, size_embedding),
                                                   name="decomposition_innput")
    decompositon_layer = layers.DenseLayer(decomposition_layer_input,
                                           num_units=size_embedding,
                                           name="decomposition")
    l_acc = RecurrentAccumulationLayer(l_Z_sum,
                                       decompositon_layer,
                                       n_steps=nb_items,
                                       name="recurrent_decoder")
    l_input_reconstruction = RealEmbeddingLayer(l_acc, size_embedding,
                                                size_items, W=l_Z.W.T,
                                                name="input_rec")
    layers_latent_to_input = [
        l_latent,
        l_acc,
        l_input_reconstruction
    ]
    return layers_input_to_latent, layers_latent_to_input
if __name__ == "__main__":
    import numpy as np
    np.random.seed(1234)
    input_to_latent, latent_to_input = build_model()

    latent = latent_to_input[0]
    input_rec = latent_to_input[2]

    H = np.random.uniform(size=(10, 100)).astype(np.float32)
    f = theano.function([latent.input_var], layers.get_output(input_rec))
    print(f(H).shape)
