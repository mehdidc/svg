import glob
from invoke import task
import os


@task
def train_supervised(mode="img_to_seq"):
    import numpy as np
    from data import read_bezier_dataset, read_images, resize_images
    from data import absolute_position, relative_position

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


def train_supervised_img_to_seq_(img, seq, img_shape):
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
                                                   kind="conv",
                                                   img_shape=img_shape)
    model = InputOutputMapping(
        [input_seq],
        [output_img]
    )
    # optimizer
    learning_rate = 0.0001
    batch_optimizer = BatchOptimizer(
        whole_dataset_in_device=False,
        batch_size=20,
        max_nb_epochs=1,
        verbose=1,
        optimization_procedure=(updates.adam,
                                {"learning_rate": learning_rate}),
    )
    # Putting everything together

    def loss_function(model, tensors):
        X_batch, y_batch = tensors["X"], tensors["y"]
        y_pred, = model.get_output(X_batch)
        return ((y_pred - y_batch) ** 2).sum(axis=1).mean()

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
    #X, y = X[0:10], y[0:10]
    X, y = X, y
    for nb in range(100000):
        i = np.random.choice(range(len(y)))
        capsule.fit(X=X[i: i + 1], y=y[i:i + 1])
        if nb % 10 == 0:
            i = np.random.choice(range(len(y)))
            y_ = capsule.predict(X[i:i+1])
            imsave("out_supervised_seq_to_img/out-{}.png".format(nb), y_[0])


def train_supervised_seq_to_img_(img, seq, img_shape):
    import numpy as np
    from collections import OrderedDict
    from lasagne import updates
    from lasagnekit.nnet.capsule import Capsule
    from lasagnekit.easy import BatchOptimizer, InputOutputMapping
    from model import build_model_img_to_seq
    from svg import gen_svg_from_output

    import theano.tensor as T
    import theano

    X = seq
    y = img

    nb_items = theano.shared(0)

    # model
    input_image, output_seq = build_model_img_to_seq(nb_items=nb_items,
                                                     kind="conv",
                                                     img_shape=img_shape)
    model = InputOutputMapping(
        [input_image],
        [output_seq]
    )
    # optimizer
    learning_rate = 0.0001
    batch_optimizer = BatchOptimizer(
        whole_dataset_in_device=False,
        batch_size=20,
        max_nb_epochs=10,
        verbose=1,
        optimization_procedure=(updates.adam,
                                {"learning_rate": learning_rate}),
    )
    # Putting everything together

    def loss_function(model, tensors):
        X_batch, y_batch = tensors["X"], tensors["y"]
        y_pred, = model.get_output(X_batch)
        return ((y_pred - y_batch) ** 2).sum(axis=1).mean()

    input_variables = OrderedDict()
    input_variables["X"] = dict(tensor_type=T.tensor4)
    input_variables["y"] = dict(tensor_type=T.tensor3)

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
    X, y = X[0:10], y[0:10]
    for nb in range(10000):
        i = np.random.choice(range(len(y)))
        nb_items.set_value(len(y[i]))
        capsule.fit(X=X[i: i + 1], y=y[i:i + 1])

        if nb % 10 == 0:
            i = np.random.choice(range(len(y)))
            y_ = capsule.predict(X[i:i+1])
            svg_content = gen_svg_from_output(y_[0])
            with open("out_supervised/out-{}.svg".format(nb), "w") as fd:
                fd.write(svg_content)


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
