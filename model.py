import theano
from lasagne import layers, nonlinearities
from helpers import (
        RealEmbeddingLayer, RecurrentAccumulationLayer, CondGRULayer,
        TensorDenseLayer)
from helpers import SumLayer, RecurrentSimpleLayer, steep_sigmoid
from lasagne.layers.recurrent import Gate
from lasagne.init import Orthogonal
from lasagne.nonlinearities import tanh, linear


import numpy as np


def build_model(batch_size=None, nb_items=5,
                size_items=8, size_embedding=100,
                size_latent=100):

    # Input to Latent
    l_input = layers.InputLayer((batch_size, nb_items, size_items),
                                name="input")
    l_Z = RealEmbeddingLayer(l_input, size_items, size_embedding,
                             name="emb")
    l_Z_sum = SumLayer(l_Z, axis=1, name="emb_sum")
    l_latent_reconstruction_mean = layers.DenseLayer(
        l_Z_sum,
        num_units=size_latent,
        nonlinearity=nonlinearities.linear,
        name="latent_rec_mean")
    l_latent_reconstruction_log_sigma = layers.DenseLayer(
        l_Z_sum,
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

    l_Z_sum = layers.DenseLayer(
        l_latent,
        num_units=size_embedding, name="latent_sum")
    decomposition_layer_input = layers.InputLayer(
        (batch_size, size_embedding*2),
        name="decomposition_input")
    decomposition_layer = layers.DenseLayer(
        decomposition_layer_input,
        num_units=size_embedding,
        nonlinearity=nonlinearities.tanh,
        name="decomposition")
    l_acc = RecurrentAccumulationLayer(
        l_Z_sum,
        decomposition_layer,
        n_steps=nb_items,
        name="recurrent_decoder")
    l_input_reconstruction = RealEmbeddingLayer(
        l_acc, size_embedding,
        size_items, W=l_Z.W.T,
        name="input_rec_pre_mean")
    l_input_reconstruction = layers.NonlinearityLayer(
        l_input_reconstruction,
        nonlinearities.tanh,
        name="input_rec_mean")
    l_input_reconstruction_log_sigma = RealEmbeddingLayer(
        l_acc, size_embedding,
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


def build_model_img_to_seq(batch_size=None,
                           img_shape=None,
                           nb_items=5,
                           nb_colors=1,
                           size_items=8,
                           kind="mlp",
                           n_out=8,
                           size_hidden=500):
    assert (img_shape is not None) and (len(img_shape) == 2)
    input_img_shape = (batch_size, nb_colors,
                       img_shape[0], img_shape[1])
    input_seq_shape = (batch_size, nb_items, size_items)

    l_img = layers.InputLayer(input_img_shape, name="img")
    l_seq = layers.InputLayer(input_seq_shape, name="seq")
    
    l_input = l_img
    if kind == "conv":
        l_conv1_1 = layers.Conv2DLayer(l_input, num_filters=32, filter_size=(3, 3),
                                       pad=1, name="conv1_1")
        l_conv2_1 = layers.Conv2DLayer(l_conv1_1, num_filters=32,
                                       filter_size=(3, 3),
                                       pad=1, name="conv2_1")
        l_pool1 = layers.MaxPool2DLayer(l_conv2_1, pool_size=(2, 2), name="pool1")
        l_conv1_2 = layers.Conv2DLayer(l_pool1, num_filters=64, filter_size=(3, 3),
                                       pad=1, name="conv1_2")
        l_conv2_2 = layers.Conv2DLayer(l_conv1_2, num_filters=64,
                                       filter_size=(3, 3),
                                       pad=1,
                                       name="conv2_2")
        l_conv3_2 = layers.Conv2DLayer(l_conv2_2, num_filters=64,
                                       filter_size=(3, 3),
                                       pad=1, name="conv3_2")
        l_pool2 = layers.MaxPool2DLayer(l_conv3_2, pool_size=(2, 2),
                                        name="pool2")
        l_hidden = layers.DenseLayer(l_pool2, size_hidden, name="code")
    elif kind == "mlp":
        l_hidden = l_input
        for i in range(3):
            l_hidden = layers.DenseLayer(l_hidden, num_units=size_hidden, name="code")
    l_pre_output = CondGRULayer(
        l_seq,
        resetgate=Gate(W_cell=None, W_hid=Orthogonal(),
                       nonlinearity=steep_sigmoid),
        updategate=Gate(W_cell=None,
                        W_hid=Orthogonal(), nonlinearity=steep_sigmoid),
        hidden_update=Gate(W_cell=None,
                           W_hid=Orthogonal(),
                           nonlinearity=tanh),
        num_units=n_out, bias=l_hidden,
        name="pre_output")
    l_output = TensorDenseLayer(l_pre_output, num_units=n_out, nonlinearity=linear, name="output")
    return l_img, l_seq, l_hidden, l_output


def build_model_seq_to_img(batch_size=None,
                           img_shape=None,
                           nb_items=5,
                           nb_colors=1,
                           size_items=8,
                           kind="mlp",
                           size_hidden=400):
    assert (img_shape is not None) and (len(img_shape) == 2)
    input_shape = (batch_size, nb_items, size_items)
    output_shape = tuple([batch_size, nb_colors] + list(img_shape))

    l_input = layers.InputLayer(input_shape, name="input")

    l_recurrent = layers.RecurrentLayer(l_input, num_units=128)
    l_recurrent = layers.RecurrentLayer(l_recurrent, num_units=128)
    l_recurrent = layers.RecurrentLayer(l_recurrent,
                                        num_units=np.prod(output_shape[1:]),
                                        nonlinearity=nonlinearities.linear)
    l_output = SumLayer(l_recurrent, axis=1)
    l_output = layers.NonlinearityLayer(l_output, nonlinearities.tanh)
    l_output = layers.ReshapeLayer(l_output, [[0]] + list(output_shape[1:]))
    return l_input, l_output


if __name__ == "__main__":
    np.random.seed(1234)

    # unsupervised
    print("Unsupervised")
    # input_to_latent, latent_to_input = build_model(size_latent=100)
    input_to_latent, latent_to_input = build_model_seq_to_seq(size_latent=100)

    input = input_to_latent[0]
    latent_rec = input_to_latent[-1]

    X = np.random.uniform(size=(10, 5, 8)).astype(np.float32)
    f = theano.function([input.input_var], layers.get_output(latent_rec))
    print(X.shape, f(X).shape)

    latent = latent_to_input[0]
    input_rec = latent_to_input[-1]

    H = np.random.uniform(size=(10, 100)).astype(np.float32)
    f = theano.function([latent.input_var], layers.get_output(input_rec))
    print(H.shape, f(H).shape)

    # supervised (img_to_seq)
    print("Supervised img_to_seq")
    size_items = 8
    input_img, input_seq, hid, output_seq = build_model_img_to_seq(img_shape=(32, 32),
                                                                   nb_items=None,
                                                                   size_items=size_items)
    X = np.random.uniform(size=(10, 1, 32, 32)).astype(np.float32)
    X_seq = np.random.uniform(size=(10, 5, size_items)).astype(np.float32)
    f = theano.function([input_img.input_var, input_seq.input_var],
                        layers.get_output(output_seq))
    print("Inputs shape : {}, {}, Output shape : {}".format(X.shape, X_seq.shape, f(X, X_seq).shape))
    # supervised (seq_to_img)
    print("Supervised seq_to_img")

    input_seq, output_img = build_model_seq_to_img(img_shape=(32, 32))
    X = np.random.uniform(size=(10, 5, 8)).astype(np.float32)
    f = theano.function([input_seq.input_var], layers.get_output(output_img))
    print(X.shape, f(X).shape)
