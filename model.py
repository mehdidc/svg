
import theano
import theano.tensor as T

from lasagne import layers, init, nonlinearities

from helpers import RealEmbeddingLayer, RecurrentAccumulationLayer
from helpers import SumLayer, RecurrentSimpleLayer, ReduceLayer


def build_model(batch_size=None, nb_items=5,
                size_items=8, size_embedding=100,
                size_latent=100):

    # Input to Latent
    l_input = layers.InputLayer((batch_size, nb_items, size_items),
                                name="input")
    l_Z = RealEmbeddingLayer(l_input, size_items, size_embedding,
                             name="emb")
    l_Z_sum = SumLayer(l_Z, axis=1, name="emb_sum")
    l_latent_reconstruction_mean = layers.DenseLayer(l_Z_sum,
                                                     num_units=size_latent,
                                                     nonlinearity=nonlinearities.linear,
                                                     name="latent_rec_mean")
    l_latent_reconstruction_log_sigma = layers.DenseLayer(l_Z_sum,
                                                          num_units=size_latent,
                                                          name="latent_rec_std",
                                                          nonlinearity=nonlinearities.linear)
    layers_input_to_latent = [
        l_input,
        l_Z,
        l_Z_sum,
        l_latent_reconstruction_mean,
        l_latent_reconstruction_log_sigma
    ]

    # Latent to Input
    l_latent = layers.InputLayer((batch_size, size_latent),
                                 name="latent")

    l_Z_sum = layers.DenseLayer(l_latent, num_units=size_embedding, name="latent_sum")
    decomposition_layer_input = layers.InputLayer((batch_size, size_embedding*2),
                                                  name="decomposition_input")
    decomposition_layer = layers.DenseLayer(decomposition_layer_input,
                                           num_units=size_embedding,
                                           nonlinearity=nonlinearities.tanh,
                                           name="decomposition")
    l_acc = RecurrentAccumulationLayer(l_Z_sum,
                                       decomposition_layer,
                                       n_steps=nb_items,
                                       name="recurrent_decoder")
    l_input_reconstruction = RealEmbeddingLayer(l_acc, size_embedding,
                                                size_items, W=l_Z.W.T,
                                                name="input_rec_pre_mean")
    l_input_reconstruction = layers.NonlinearityLayer(l_input_reconstruction,
                                                      nonlinearities.tanh,
                                                      name="input_rec_mean")
    l_input_reconstruction_log_sigma = RealEmbeddingLayer(l_acc, size_embedding,
                                                          size_items,
                                                          name="input_rec_std")

    layers_latent_to_input = [
        l_latent,
        l_acc,
        l_input_reconstruction,
        l_input_reconstruction_log_sigma
    ]
    return layers_input_to_latent, layers_latent_to_input

def build_model_seq_to_seq(batch_size=None, nb_items=5,
                           size_items=8,
                           size_hidden=400,
                           size_latent=400):
    l_input = layers.InputLayer((batch_size, None, size_items),
                                 name="input")
    l_hidden = layers.LSTMLayer(l_input, num_units=size_hidden, name="hidden")
    l_hidden_sliced = layers.SliceLayer(l_hidden, indices=-1, axis=1, name="hidden_sliced")
    l_latent_reconstruction_mean = layers.DenseLayer(l_hidden_sliced,
                                                     num_units=size_latent,
                                                     nonlinearity=nonlinearities.linear,
                                                     name="latent_rec_mean")
    l_latent_reconstruction_sigma = layers.DenseLayer(l_hidden_sliced,
                                                      num_units=size_latent,
                                                      name="latent_rec_std",
                                                      nonlinearity=nonlinearities.linear)
    layers_input_to_latent = [
        l_input,
        l_hidden,
        l_hidden_sliced,
        l_latent_reconstruction_mean,
        l_latent_reconstruction_sigma
    ]

    l_latent = layers.InputLayer((batch_size, size_latent),
                                 name="latent")
    l_initial_hidden = layers.DenseLayer(l_latent, size_hidden, name="initial_hidden")

    decomposition_layer_input = layers.InputLayer((batch_size, size_hidden))
    decomposition_layer = layers.DenseLayer(decomposition_layer_input,
                                            num_units=size_hidden,
                                            name="decomposition_layer")

    l_hidden = RecurrentSimpleLayer(l_initial_hidden, decomposition_layer,
                                    n_steps=nb_items, name="hidden")
    l_input_reconstruction = layers.LSTMLayer(l_hidden,
                                              num_units=size_items * 2,
                                              nonlinearity=nonlinearities.linear,
                                              name="input_rec")
    l_input_reconstruction_mean = layers.SliceLayer(l_input_reconstruction,
                                                    indices=slice(0, size_items),
                                                    axis=2,
                                                    name="input_rec_mean")
    l_input_reconstruction_std = layers.SliceLayer(l_input_reconstruction,
                                                   indices=slice(size_items, -1),
                                                   axis=2,
                                                   name="input_rec_std")

    layers_latent_to_input = [
        l_latent,
        l_initial_hidden,
        l_hidden,
        decomposition_layer,
        l_input_reconstruction_mean,
        l_input_reconstruction_std
    ]

    return layers_input_to_latent, layers_latent_to_input

if __name__ == "__main__":
    import numpy as np
    np.random.seed(1234)

    #input_to_latent, latent_to_input = build_model()
    input_to_latent, latent_to_input = build_model_seq_to_seq()

    input = input_to_latent[0]
    latent_rec = input_to_latent[-1]

    X = np.random.uniform(size=(10, 5, 8)).astype(np.float32)
    f = theano.function([input.input_var], layers.get_output(latent_rec))
    print(f(X).shape)

    latent = latent_to_input[0]
    input_rec = latent_to_input[-1]

    H = np.random.uniform(size=(10, 100)).astype(np.float32)
    f = theano.function([latent.input_var], layers.get_output(input_rec))
    print(f(H).shape)
