import theano
import theano.tensor as T

from lasagne import layers, init, nonlinearities

from collections import defaultdict

class ReduceLayer(layers.Layer):
    def __init__(self, incoming,
                 reduce_function,
                 axis=-1,
                 **kwargs):
        super(ReduceLayer, self).__init__(incoming, **kwargs)
        self.axis = axis

    def get_output_for(self, input, **kwargs):
        return self.reduce_function(input)

    def get_output_shape_for(self, input_shape):
        shape = (list(input_shape[0:self.axis]) +
                 list(input_shape[self.axis + 1:]))
        shape = tuple(shape)
        return shape

class SumLayer(ReduceLayer):

    def __init__(self, incoming, axis=-1, **kwargs):
        reduce_function = lambda x, axis: x.mean(axis=axis)
        super(SumLayer, self).__init__(incoming, reduce_function , axis=axis,
                                       **kwargs)

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
        input_i_concat = T.concatenate((X.dimshuffle(0, 'x', 1), input_i.dimshuffle(0, 'x', 1)), axis=1)
        delta_input = layers.get_output(decomposition_layer, input_i_concat)
        output = nonlinearities.tanh(input_i - delta_input )
        return output

    sequences = []
    outputs_info = [T.zeros(X.shape)]

    non_sequences = layers.get_all_params(decomposition_layer) + [X]

    result, updates = theano.scan(fn=step_function,
                                  sequences=sequences,
                                  outputs_info=outputs_info,
                                  non_sequences=non_sequences,
                                  strict=True,
                                  n_steps=n_steps,
                                  **scan_kwargs)
    result = result.dimshuffle(1, 0, 2)
    return result, updates


class RecurrentSimpleLayer(layers.Layer):

    def __init__(self, incoming, decomposition_layer, n_steps, name=None):
        super(RecurrentSimpleLayer, self).__init__(incoming,
                                                   name=name)
        self.decomposition_layer = decomposition_layer
        self.n_steps = n_steps

    def get_output_for(self, input, **kwargs):
        result, updates = recurrent_simple(input,
                                           self.decomposition_layer,
                                           self.n_steps)
        return result

    def get_output_shape_for(self, input_shape):
        shape_one_step = self.decomposition_layer.get_output_shape_for(input_shape)

        theano_stuff =  (isinstance(self.n_steps, T.TensorVariable) or
                         isinstance(self.n_steps, T.sharedvar.SharedVariable))
        if theano_stuff:
            n_steps = None
        else:
            n_steps = self.n_steps
        shape = [shape_one_step[0], n_steps] + list(shape_one_step[1:])
        shape = tuple(shape)
        return shape


def recurrent_simple(X, decomposition_layer, n_steps,
                     **scan_kwargs):

    def step_function(hidden_i, *args):
        hidden_next = layers.get_output(decomposition_layer, hidden_i)
        return hidden_next

    sequences = []
    outputs_info = [X]

    non_sequences = layers.get_all_params(decomposition_layer)

    result, updates = theano.scan(fn=step_function,
                                  sequences=sequences,
                                  outputs_info=outputs_info,
                                  non_sequences=non_sequences,
                                  strict=True,
                                  n_steps=n_steps,
                                  **scan_kwargs)
    result = result.dimshuffle(1, 0, 2)
    return result, updates


def iterate_over_variable_size_minibatches(X, axis=0, nb_epochs=1):
    per_size = defaultdict(list)
    for x in X:
        per_size[x.shape[axis]].append(x)
    print(per_size.keys())
    for i in range(nb_epochs):
        for size, content in per_size.items():
            yield content
