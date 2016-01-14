import theano
import numpy as np
import theano.tensor as T
from lasagne.layers import MergeLayer, Layer
from lasagne.layers.recurrent import Gate
from lasagne import init, nonlinearities
import theano.tensor as T
from lasagne.utils import unroll_scan


from lasagne import layers, init, nonlinearities

from collections import defaultdict


class ReduceLayer(layers.Layer):
    def __init__(self, incoming,
                 reduce_function,
                 axis=-1,
                 **kwargs):
        super(ReduceLayer, self).__init__(incoming, **kwargs)
        self.reduce_function = reduce_function
        self.axis = axis

    def get_output_for(self, input, **kwargs):
        return self.reduce_function(input, self.axis)

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
    for i in range(nb_epochs):
        for size, content in per_size.items():
            yield content

class CondGRULayer(MergeLayer):
    r"""
    lasagne.layers.recurrent.GRULayer(incoming, num_units,
    resetgate=lasagne.layers.Gate(W_cell=None),
    updategate=lasagne.layers.Gate(W_cell=None),
    hidden_update=lasagne.layers.Gate(
    W_cell=None, lasagne.nonlinearities.tanh),
    hid_init=lasagne.init.Constant(0.), backwards=False, learn_init=False,
    gradient_steps=-1, grad_clipping=0, unroll_scan=False,
    precompute_input=True, mask_input=None, only_return_final=False, **kwargs)

    Gated Recurrent Unit (GRU) Layer

    Implements the recurrent step proposed in [1]_, which computes the output
    by

    .. math ::
        r_t &= \sigma_r(x_t W_{xr} + h_{t - 1} W_{hr} + b_r)\\
        u_t &= \sigma_u(x_t W_{xu} + h_{t - 1} W_{hu} + b_u)\\
        c_t &= \sigma_c(x_t W_{xc} + r_t \odot (h_{t - 1} W_{hc}) + b_c)\\
        h_t &= (1 - u_t) \odot h_{t - 1} + u_t \odot c_t

    Parameters
    ----------
    incoming : a :class:`lasagne.layers.Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    num_units : int
        Number of hidden units in the layer.
    resetgate : Gate
        Parameters for the reset gate (:math:`r_t`): :math:`W_{xr}`,
        :math:`W_{hr}`, :math:`b_r`, and :math:`\sigma_r`.
    updategate : Gate
        Parameters for the update gate (:math:`u_t`): :math:`W_{xu}`,
        :math:`W_{hu}`, :math:`b_u`, and :math:`\sigma_u`.
    hidden_update : Gate
        Parameters for the hidden update (:math:`c_t`): :math:`W_{xc}`,
        :math:`W_{hc}`, :math:`b_c`, and :math:`\sigma_c`.
    hid_init : callable, np.ndarray, theano.shared or TensorVariable
        Initializer for initial hidden state (:math:`h_0`).  If a
        TensorVariable (Theano expression) is supplied, it will not be learned
        regardless of the value of `learn_init`.
    backwards : bool
        If True, process the sequence backwards and then reverse the
        output again such that the output from the layer is always
        from :math:`x_1` to :math:`x_n`.
    learn_init : bool
        If True, initial hidden values are learned. If `hid_init` is a
        TensorVariable then the TensorVariable is used and
        `learn_init` is ignored.
    gradient_steps : int
        Number of timesteps to include in the backpropagated gradient.
        If -1, backpropagate through the entire sequence.
    grad_clipping : float
        If nonzero, the gradient messages are clipped to the given value during
        the backward pass.  See [1]_ (p. 6) for further explanation.
    unroll_scan : bool
        If True the recursion is unrolled instead of using scan. For some
        graphs this gives a significant speed up but it might also consume
        more memory. When `unroll_scan` is True, backpropagation always
        includes the full sequence, so `gradient_steps` must be set to -1 and
        the input sequence length must be known at compile time (i.e., cannot
        be given as None).
    precompute_input : bool
        If True, precompute input_to_hid before iterating through
        the sequence. This can result in a speedup at the expense of
        an increase in memory usage.
    mask_input : :class:`lasagne.layers.Layer`
        Layer which allows for a sequence mask to be input, for when sequences
        are of variable length.  Default `None`, which means no mask will be
        supplied (i.e. all sequences are of the same length).
    only_return_final : bool
        If True, only return the final sequential output (e.g. for tasks where
        a single target value for the entire sequence is desired).  In this
        case, Theano makes an optimization which saves memory.

    References
    ----------
    .. [1] Cho, Kyunghyun, et al: On the properties of neural
       machine translation: Encoder-decoder approaches.
       arXiv preprint arXiv:1409.1259 (2014).
    .. [2] Chung, Junyoung, et al.: Empirical Evaluation of Gated
       Recurrent Neural Networks on Sequence Modeling.
       arXiv preprint arXiv:1412.3555 (2014).
    .. [3] Graves, Alex: "Generating sequences with recurrent neural networks."
           arXiv preprint arXiv:1308.0850 (2013).

    Notes
    -----
    An alternate update for the candidate hidden state is proposed in [2]_:

    .. math::
        c_t &= \sigma_c(x_t W_{ic} + (r_t \odot h_{t - 1})W_{hc} + b_c)\\

    We use the formulation from [1]_ because it allows us to do all matrix
    operations in a single dot product.
    """
    def __init__(self,
                 incoming,
                 num_units,
                 bias,
                 resetgate=Gate(W_cell=None),
                 updategate=Gate(W_cell=None),
                 hidden_update=Gate(W_cell=None,
                                    nonlinearity=nonlinearities.tanh),
                 hid_init=init.Constant(0.),
                 Cr=init.GlorotUniform(),
                 Cu=init.GlorotUniform(),
                 Ch=init.GlorotUniform(),
                 backwards=False,
                 learn_init=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=False,
                 **kwargs):

        # This layer inherits from a MergeLayer, because it can have two
        # inputs - the layer input, and the mask.  We will just provide the
        # layer input as incomings, unless a mask input was provided.
        incomings = [incoming, bias]
        if mask_input is not None:
            incomings.append(mask_input)

        # Initialize parent layer
        super(CondGRULayer, self).__init__(incomings, **kwargs)

        self.learn_init = learn_init
        self.num_units = num_units
        self.grad_clipping = grad_clipping
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input
        self.only_return_final = only_return_final

        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")

        # Retrieve the dimensionality of the incoming layer
        input_shape = self.input_shapes[0]

        if unroll_scan and input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")

        # Input dimensionality is the output dimensionality of the input layer
        num_inputs = np.prod(input_shape[2:])

        def add_gate_params(gate, gate_name):
            """ Convenience function for adding layer parameters from a Gate
            instance. """
            return (self.add_param(gate.W_in, (num_inputs, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(gate.W_hid, (num_units, num_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    None,
                    #self.add_param(gate.b, (num_units,),
                    #               name="b_{}".format(gate_name),
                    #               regularizable=False),
                    gate.nonlinearity)

        # Add in all parameters from gates
        (self.W_in_to_updategate, self.W_hid_to_updategate, self.b_updategate,
         self.nonlinearity_updategate) = add_gate_params(updategate,
                                                         'updategate')
        (self.W_in_to_resetgate, self.W_hid_to_resetgate, self.b_resetgate,
         self.nonlinearity_resetgate) = add_gate_params(resetgate, 'resetgate')

        (self.W_in_to_hidden_update, self.W_hid_to_hidden_update,
         self.b_hidden_update, self.nonlinearity_hid) = add_gate_params(
             hidden_update, 'hidden_update')
        num_bias = bias.output_shape[1]
        self.Cr = self.add_param(Cr, (num_bias, num_units))
        self.Cu = self.add_param(Cu, (num_bias, num_units))
        self.Ch = self.add_param(Ch, (num_bias, num_units))

        # Initialize hidden state
        if isinstance(hid_init, T.TensorVariable):
            if hid_init.ndim != 2:
                raise ValueError(
                    "When hid_init is provided as a TensorVariable, it should "
                    "have 2 dimensions and have shape (num_batch, num_units)")
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1, self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False)

    def get_output_shape_for(self, input_shapes):
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shape = input_shapes[0]
        # When only_return_final is true, the second (sequence step) dimension
        # will be flattened
        if self.only_return_final:
            return input_shape[0], self.num_units
        # Otherwise, the shape will be (n_batch, n_steps, num_units)
        else:
            return input_shape[0], input_shape[1], self.num_units

    def get_output_for(self, inputs, **kwargs):
        """
        Compute this layer's output function given a symbolic input variable

        Parameters
        ----------
        inputs : list of theano.TensorType
            `inputs[0]` should always be the symbolic input variable.  When
            this layer has a mask input (i.e. was instantiated with
            `mask_input != None`, indicating that the lengths of sequences in
            each batch vary), `inputs` should have length 2, where `inputs[1]`
            is the `mask`.  The `mask` should be supplied as a Theano variable
            denoting whether each time step in each sequence in the batch is
            part of the sequence or not.  `mask` should be a matrix of shape
            ``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
            (length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
            of sequence i)``.

        Returns
        -------
        layer_output : theano.TensorType
            Symbolic output variable.
        """
        # Retrieve the layer input
        input = inputs[0]
        bias = inputs[1]
        # Retrieve the mask when it is supplied
        mask = inputs[2] if len(inputs) > 2 else None


        #  Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = T.flatten(input, 3)

        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = input.shape

        # Stack input weight matrices into a (num_inputs, 3*num_units)
        # matrix, which speeds up computation
        W_in_stacked = T.concatenate(
            [self.W_in_to_resetgate, self.W_in_to_updategate,
             self.W_in_to_hidden_update], axis=1)

        # Same for hidden weight matrices
        W_hid_stacked = T.concatenate(
            [self.W_hid_to_resetgate, self.W_hid_to_updategate,
             self.W_hid_to_hidden_update], axis=1)

        # Stack gate biases into a (3*num_units) vector
        
        #b_stacked = T.concatenate(
        #    [self.b_resetgate, self.b_updategate,
        #     self.b_hidden_update], axis=0)
        b_stacked = None

        b_r = T.dot(bias, self.Cr)
        #b_r = theano.printing.Print("")(b_r)
        b_u = T.dot(bias, self.Cu)
        #b_r = theano.printing.Print("")(b_u)
        b_h = T.dot(bias, self.Ch)

        if self.precompute_input:
            # precompute_input inputs*W. W_in is (n_features, 3*num_units).
            # input is then (n_batch, n_time_steps, 3*num_units).
            input = T.dot(input, W_in_stacked)# + b_stacked

        # At each call to scan, input_n will be (n_time_steps, 3*num_units).
        # We define a slicing function that extract the input to each GRU gate
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]

        # Create single recurrent computation step function
        # input__n is the n'th vector of the input
        def step(input_n, hid_previous, *args):
            # Compute W_{hr} h_{t - 1}, W_{hu} h_{t - 1}, and W_{hc} h_{t - 1}
            hid_input = T.dot(hid_previous, W_hid_stacked)

            if self.grad_clipping:
                input_n = theano.gradient.grad_clip(
                    input_n, -self.grad_clipping, self.grad_clipping)
                hid_input = theano.gradient.grad_clip(
                    hid_input, -self.grad_clipping, self.grad_clipping)

            if not self.precompute_input:
                # Compute W_{xr}x_t + b_r, W_{xu}x_t + b_u, and W_{xc}x_t + b_c
                input_n = T.dot(input_n, W_in_stacked) #+ b_stacked

            # Reset and update gates
            resetgate = slice_w(hid_input, 0) + slice_w(input_n, 0) + b_r
            updategate = slice_w(hid_input, 1) + slice_w(input_n, 1) + b_u
            resetgate = self.nonlinearity_resetgate(resetgate)
            updategate = self.nonlinearity_updategate(updategate)

            # Compute W_{xc}x_t + r_t \odot (W_{hc} h_{t - 1})
            hidden_update_in = slice_w(input_n, 2)
            hidden_update_hid = slice_w(hid_input, 2)
            hidden_update = hidden_update_in + resetgate*hidden_update_hid + b_h
            if self.grad_clipping:
                hidden_update = theano.gradient.grad_clip(
                    hidden_update, -self.grad_clipping, self.grad_clipping)
            hidden_update = self.nonlinearity_hid(hidden_update)

            # Compute (1 - u_t)h_{t - 1} + u_t c_t
            hid = (1 - updategate)*hid_previous + updategate*hidden_update
            return hid

        def step_masked(input_n, mask_n, hid_previous, *args):
            hid = step(input_n, hid_previous, *args)

            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            not_mask = 1 - mask_n
            hid = hid*mask_n + hid_previous*not_mask

            return hid

        if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
        else:
            sequences = [input]
            step_fun = step

        if isinstance(self.hid_init, T.TensorVariable):
            hid_init = self.hid_init
        else:
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(T.ones((num_batch, 1)), self.hid_init)

        # The hidden-to-hidden weight matrix is always used in step
        non_seqs = [W_hid_stacked, b_r, b_u, b_h]
        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        if not self.precompute_input:
            non_seqs += [W_in_stacked, b_stacked]

        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            hid_out = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[hid_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])[0]
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            hid_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                go_backwards=self.backwards,
                outputs_info=[hid_init],
                non_sequences=non_seqs,
                truncate_gradient=self.gradient_steps,
                strict=True)[0]

        # When it is requested that we only return the final sequence step,
        # we need to slice it out immediately after scan is applied
        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]

        return hid_out


class Repeat(Layer):
    def __init__(self, incoming, n, **kwargs):
        super(Repeat, self).__init__(incoming, **kwargs)
        self.n = n

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.n) + input_shape[1:]

    def get_output_for(self, input, **kwargs):
        input = input[:, None, :]
        return T.extra_ops.repeat(input, self.n, axis=1)


def gaussian_ll(X, x_mean_hat, x_log_sigma_hat):
    """
    n_in = 8
    n_hid = 800
    size_mixture = 20
    n_out = 8 * 2 * size_mixture + size_mixture

    def extract_from_y_dist(y_dist):
        y_dist_mixtures =    y_dist[:, :, 0:size_mixture*8*2]
        y_dist_mixtures = y_dist_mixtures.reshape((y_dist_mixtures.shape[0], 
                                                   y_dist_mixtures.shape[1], 
                                                   size_mixture, 2, 8))
        y_mean = T.tanh(y_dist_mixtures[:, :, :, 0, :])
        y_std = T.exp(y_dist_mixtures[:, :, :, 1, :])
        
        #100, 25, 7, 8
        y_dist_proportions = y_dist[:, :, size_mixture*8*2:]
        y_dist_proportions = y_dist_proportions.reshape((y_dist_proportions.shape[0],
                                                         y_dist_proportions.shape[1],
                                                         size_mixture,
                                                        ))
    # 100, 25, 7
    y_dist_proportions = softmax(y_dist_proportions, axis=2)
    return y_mean, y_std, y_dist_proportions

    y_mean, y_std, y_dist_proportions = extract_from_y_dist(y_dist)
    y_ = ty.dimshuffle(0, 1, 'x', 2)
    #100, 25, 7
    per_mixture = gaussian_ll(y_, y_mean, y_std).sum(axis=3) + T.log(y_dist_proportions)
    #100, 25, 7
    per_example_and_time = log_sum_exp(per_mixture, axis=2)
    per_example = per_example_and_time.sum(axis=1).mean()
    """
    # x_mean_hat and x_sigma_hat computed by a neural network
    x_hat = X - x_mean_hat
    ll = -0.5 * ((x_hat ** 2) / T.exp(2. * x_log_sigma_hat) + 2 * x_log_sigma_hat) - 0.5*np.log(2*np.pi)
    return ll


def softmax(x, axis=1, backend=T):
    e_x = backend.exp(x - x.max(axis=axis, keepdims=True))
    out = e_x / e_x.sum(axis=axis, keepdims=True)
    return out


def steep_sigmoid(x):
    return nonlinearities.sigmoid(x*3.75)


class TensorDenseLayer(Layer):
    """
    used to perform embeddings on arbitray input tensor
    X : ([0], [1], ...,  T)
    W : (T, E) where E is the embedding size and T is last dim input size
    returns tensordot(X, W) + b which is : ([0], [1], ..., E)
    """
    def __init__(self, incoming, num_units, W=init.GlorotUniform(),
                 b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
                 **kwargs):
        super(TensorDenseLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

        self.num_units = num_units
        num_inputs = self.input_shape[-1]
        self.W = self.add_param(W, (num_inputs, num_units), name="W")
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_units,), name="b",
                                    regularizable=False)

    def get_output_shape_for(self, input_shape):
        return input_shape[0:-1] + (self.num_units,)

    def get_output_for(self, input, **kwargs):
        activation = T.tensordot(input, self.W, axes=[(input.ndim - 1,), (0,)])
        if self.b is not None:
            activation = activation + self.b
        return self.nonlinearity(activation)


