import matplotlib as mpl
mpl.use('Agg')  # NOQA
import matplotlib.pyplot as plt
import numpy as np
import glob
from invoke import task
import os


@task
def train_supervised(mode="img_to_seq"):
    import numpy as np
    from data import read_bezier_dataset, read_images, resize_images

    state = 1234
    np.random.seed(state)

    # data
    img_shape = (32, 32)
    filenames = glob.glob("svg/*.txt")
    X = read_images(filenames)
    X = resize_images(X, img_shape)
    X = np.array(X).astype(np.float32)
    X = X[:, None, :, :]  # 1color channel
    X = 1 - X

    y = read_bezier_dataset(filenames)

    img = X
    seq = y

    if mode == "img_to_seq":
        train_supervised_img_to_seq_(img, seq, img_shape)
    elif mode == "seq_to_img":
        train_supervised_seq_to_img_(img, seq, img_shape)
    else:
        print("Wrong mode : {}".format(mode))


def train_supervised_seq_to_img_(img, seq, img_shape):
    from collections import OrderedDict
    from lasagne import updates
    from lasagnekit.nnet.capsule import Capsule
    from lasagnekit.easy import BatchOptimizer, InputOutputMapping
    from model import build_model_seq_to_img
    import theano.tensor as T
    import numpy as np
    from skimage.io import imsave

    X = seq
    y = img
    input_seq, output_img = build_model_seq_to_img(nb_items=None,
                                                   kind="mlp",
                                                   img_shape=img_shape)
    model = InputOutputMapping(
        [input_seq],
        [output_img]
    )
    # optimizer
    learning_rate = 0.001
    batch_optimizer = BatchOptimizer(
        whole_dataset_in_device=False,
        batch_size=20,
        max_nb_epochs=1,
        verbose=0,
        optimization_procedure=(updates.nesterov_momentum,
                                {"learning_rate": learning_rate}),
    )
    # Putting everything together

    def loss_function(model, tensors):
        X_batch, y_batch = tensors["X"], tensors["y"]
        y_pred, = model.get_output(X_batch)
        return ((y_pred - y_batch) ** 2).sum(axis=2).mean()

    input_variables = OrderedDict()
    input_variables["X"] = dict(tensor_type=T.tensor3)
    input_variables["y"] = dict(tensor_type=T.tensor4)

    def predict(model, X):
        y_pred, = model.get_output(X)
        return y_pred
    functions = dict(
        predict=dict(get_output=predict, params=["X"])
    )

    capsule = Capsule(input_variables, model,
                      loss_function,
                      functions=functions,
                      batch_optimizer=batch_optimizer)
    X, y = X, y

    # X,  y = X[0:1], y[0:1]
    avg = 0
    avgs = []
    for nb in range(100000):
        i = np.random.choice(range(len(y)))

        a = X[i: i + 1]
        b = y[i: i + 1]
        capsule.fit(X=a, y=b)
        loss = ((capsule.predict(a) - b) ** 2).sum(axis=2).mean()

        B = 0.9
        t = nb
        avg = B * avg + (1 - B) * loss
        fix = 1 - B ** (1 + t)
        avg_fix = avg / fix
        avgs.append(avg_fix)

        if nb % 200 == 0:
            i = np.random.choice(range(len(y)))
            y_ = capsule.predict(X[i:i+1])
            imsave("out_supervised_seq_to_img/out-{}.png".format(nb), y_[0])
            print(avg_fix)
            fig = plt.figure()
            plt.plot(np.arange(len(avgs)), avgs)
            plt.savefig("out_supervised_seq_to_img/loss.png")
            plt.close(fig)


def train_supervised_img_to_seq_(img, seq, img_shape):
    import numpy as np
    from collections import OrderedDict
    from lasagne import updates
    from lasagnekit.nnet.capsule import Capsule
    from lasagnekit.easy import BatchOptimizer, InputOutputMapping
    from helpers import softmax
    from lasagnekit.easy import log_sum_exp
    from model import build_model_img_to_seq
    from svg import gen_svg_from_output
    from helpers import gaussian_ll
    import theano.tensor as T
    from lasagne import layers
    import theano

    size_mixture = 128
    n_out = 8 * 2 * size_mixture + size_mixture

    X_img = img
    X_seq = [s[0:-1] for s in seq]
    y_seq = [s[1:] for s in seq]

    # model
    input_img, input_seq, code, output_seq = build_model_img_to_seq(
            nb_items=None,
            nb_colors=1,
            kind='mlp',
            size_hidden=256,
            n_out=n_out,
            img_shape=img_shape)

    model = InputOutputMapping(
        [input_img, input_seq],
        [output_seq]
    )
    # optimizer
    learning_rate = 0.001
    batch_optimizer = BatchOptimizer(
        whole_dataset_in_device=False,
        batch_size=20,
        max_nb_epochs=10,
        verbose=0,
        optimization_procedure=(updates.nesterov_momentum,
            {"learning_rate": learning_rate, "momentum": 0.95}),
    )
    # Putting everything together
    def extract_from_y_dist(y_dist, backend=theano.tensor):
        y_dist_mixtures =  y_dist[:, :, 0:size_mixture*8*2]
        y_dist_mixtures = y_dist_mixtures.reshape((y_dist_mixtures.shape[0],
                                                   y_dist_mixtures.shape[1],
                                                   size_mixture, 2, 8))
        y_mean = backend.tanh(y_dist_mixtures[:, :, :, 0, :])
        y_std = backend.exp(y_dist_mixtures[:, :, :, 1, :])
        
        #100, 25, 7, 8
        y_dist_proportions = y_dist[:, :, size_mixture*8*2:]
        y_dist_proportions = y_dist_proportions.reshape((y_dist_proportions.shape[0],
                                                         y_dist_proportions.shape[1],
                                                         size_mixture,
                                                        ))
        y_dist_proportions = softmax(y_dist_proportions, axis=2, backend=backend)
        return y_mean, y_std, y_dist_proportions

    def loss_function(model, tensors):
        X_img, X_seq, y_seq = tensors["X_img"], tensors["X_seq"], tensors["y_seq"]
        y_dist, = model.get_output(X_img, X_seq)

        y_mean, y_std, y_dist_proportions = extract_from_y_dist(y_dist)
        y_real = y_seq.dimshuffle(0, 1, 'x', 2)
        #100, 25, 7
        per_mixture = (
                gaussian_ll(y_real, y_mean, y_std).sum(axis=3) +
                T.log(y_dist_proportions)
        )
        #100, 25, 7
        per_example_and_time = log_sum_exp(per_mixture, axis=2)
        per_example = per_example_and_time.mean()
        return -per_example

    input_variables = OrderedDict()
    input_variables["X_img"] = dict(tensor_type=T.tensor4)
    input_variables["X_seq"] = dict(tensor_type=T.tensor3)
    input_variables["y_seq"] = dict(tensor_type=T.tensor3)

    def predict(model, X_img, X_seq):
        y_pred, = model.get_output(X_img, X_seq)
        return y_pred

    def predict_img_repr(model, X_img):
        img_repr = layers.get_output(code, X_img)
        return img_repr

    functions = dict(
        predict=dict(get_output=predict, params=["X_img", "X_seq"]),
        predict_img_repr=dict(get_output=predict_img_repr, params=["X_img"]),
    )

    capsule = Capsule(input_variables, model,
                      loss_function,
                      functions=functions,
                      batch_optimizer=batch_optimizer)

    x_code = T.matrix()
    x_seq = T.tensor3()
    capsule.predict_given_img_repr = theano.function(
            [x_code, x_seq],
            layers.get_output(output_seq, {code: x_code, input_seq: x_seq}))
    avg = 0
    avgs = []


    for nb in range(1000000):
        #i = np.random.choice(range(len(X_img)))
        i = nb % 10
        # i = 0
        a = X_img[i: i + 1]
        b = X_seq[i: i + 1]
        c = y_seq[i: i + 1]
        capsule.fit(X_img=a, X_seq=b, y_seq=c)
        status = capsule.batch_optimizer.stats[-1]

        loss = status["loss_train"]
        B = 0.9
        t = nb
        avg = B * avg + (1 - B) * loss
        fix = 1 - B ** (1 + t)
        avg_fix = avg / fix
        avgs.append(avg_fix)

        if nb % 50 == 0:
            print(avg_fix)
            fig = plt.figure()
            plt.plot(np.arange(len(avgs)), avgs)
            plt.savefig("out_supervised_img_to_seq/loss.png")
            plt.close(fig)

            i = np.random.choice(range(len(X_img)))
            code = capsule.predict_img_repr(X_img[i:i+1])

            def predict_dist(seq):
                y_dist = capsule.predict_given_img_repr(code, seq)
                return extract_from_y_dist(y_dist, backend=np)

            nb_steps = len(y_seq[i])
            y_gen = generate(predict_dist, nb_steps=nb_steps)
            svg_content = gen_svg_from_output(y_gen)
            with open("out_supervised_img_to_seq/out-{}.svg".format(nb), "w") as fd:
                fd.write(svg_content)


def generate(predict_dist,
             initial=None, nb_steps=1, rng=np.random,
             temperature=1):
    if initial is None:
        initial = np.zeros((8,)).astype(np.float32)
    generated = np.zeros((1, nb_steps + 1, 8)).astype(np.float32)
    generated[0, 0] = initial
    # (one example, one time step, 8)
    for i in range(1, nb_steps):
        y_mean, y_std, y_dist_proportions = predict_dist(generated[:, 0:i, :])
        y_dist_proportions = y_dist_proportions[0, i - 1] ** temperature
        mixture_id = y_dist_proportions.argmax()
        y_dist_proportions[-1] = 1 - y_dist_proportions[0:-1].sum()
        y_mean = y_mean[0, i - 1, mixture_id]
        y_std = y_std[0, i - 1, mixture_id]
        y_gen = np.random.multivariate_normal(y_mean, (np.diag(y_std**2)))
        generated[0, i, :] = y_gen
    return generated[0]


@task
def train_unsupervised():
    from lasagne import updates
    from lasagnekit.generative.va import VariationalAutoencoder, Real
    from lasagnekit.generative.autoencoder import Autoencoder
    from lasagnekit.easy import BatchOptimizer, InputOutputMapping
    from lasagnekit.easy import layers_from_list_to_dict
    from data import read_bezier_dataset
    from model import build_model_seq_to_seq
    from svg import gen_svg_from_output

    import theano.tensor as T
    import theano
    from theano.sandbox import rng_mrg
    import numpy as np
    from helpers import iterate_over_variable_size_minibatches

    state = 1234
    np.random.seed(state)

    # data
    filenames = glob.glob("svg/*.txt")
    X = read_bezier_dataset(filenames)

    nb_items = theano.shared(0)

    # model
    input_to_latent, latent_to_input = build_model_seq_to_seq(nb_items=nb_items)
    input_to_latent = layers_from_list_to_dict(input_to_latent)
    latent_to_input = layers_from_list_to_dict(latent_to_input)

    input_to_latent_mapping = InputOutputMapping(
        [input_to_latent["input"]],

        #[input_to_latent["latent_rec_mean"], input_to_latent["latent_rec_std"]]
        [input_to_latent["latent_rec_mean"]]
    )

    latent_to_input_mapping = InputOutputMapping(
        [latent_to_input["latent"]],

        # [latent_to_input["input_rec_mean"], latent_to_input["input_rec_std"]]
        [latent_to_input["input_rec_mean"]]
    )

    # optimizer
    learning_rate = 0.001
    batch_optimizer = BatchOptimizer(
        whole_dataset_in_device=False,
        batch_size=20,
        max_nb_epochs=1,
        verbose=1,
        optimization_procedure=(updates.rmsprop,
                                {"learning_rate": learning_rate}),
    )

    # Putting all together
    vae = VariationalAutoencoder(input_to_latent_mapping, #NOQA
                                 latent_to_input_mapping,
                                 batch_optimizer,
                                 rng=rng_mrg.MRG_RandomStreams(seed=state),
                                 input_type=Real,
                                 X_type=T.tensor3,
                                 nb_z_samples=10)
    aa = Autoencoder(input_to_latent_mapping, #NOQA
                     latent_to_input_mapping,
                     batch_optimizer=batch_optimizer,
                     X_type=T.tensor3)
    # training
    epoch = 0
    for x in iterate_over_variable_size_minibatches(X, axis=0, nb_epochs=10000):
        x = np.array(x)
        nb_items.set_value(x.shape[1])
        aa.fit(x)
        if epoch % 100 == 0:
            h, = aa.encode(x)
            samples, = aa.decode(h)
            try:
                os.mkdir("epoch{}".format(epoch))
            except OSError:
                pass
            for s_i, s in enumerate(samples):
                svg_content = gen_svg_from_output(s)
                filename = "epoch{}/{}".format(epoch, s_i)
                with open("{}.svg".format(filename), "w") as fd:
                    fd.write(svg_content)
        epoch += 1


    # sampling
    #samples = vae.sample(nb=1)
    h,=aa.encode(X[0:1])
    samples,=aa.decode(h)
    svg_content = gen_svg_from_output(samples[0])
    with open("out.svg", "w") as fd:
        fd.write(svg_content)

if __name__ == "__main__":
    train()
