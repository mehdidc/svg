import glob
from invoke import task


@task
def train():
    from lasagne import updates
    from lasagnekit.generative.va import VariationalAutoencoder, Real, Binary
    from lasagnekit.generative.autoencoder import Autoencoder
    from lasagnekit.easy import BatchOptimizer, InputOutputMapping
    from lasagnekit.easy import layers_from_list_to_dict
    from data import read_bezier_dataset
    from model import build_model, build_model_seq_to_seq
    from svg import gen_svg_from_output

    import theano.tensor as T
    from theano.sandbox import rng_mrg
    import numpy as np

    state = 1234
    np.random.seed(state)

    # data
    filenames = glob.glob("svg/*.txt")
    X = read_bezier_dataset(filenames)

    # model
    input_to_latent, latent_to_input = build_model_seq_to_seq(nb_items=29)
    input_to_latent = layers_from_list_to_dict(input_to_latent)
    latent_to_input = layers_from_list_to_dict(latent_to_input)

    input_to_latent_mapping = InputOutputMapping(
        [input_to_latent["input"]],

        #[input_to_latent["latent_rec_mean"], input_to_latent["latent_rec_std"]]
        [input_to_latent["latent_rec_mean"]]
    )

    latent_to_input_mapping = InputOutputMapping(
        [latent_to_input["latent"]],

        #[latent_to_input["input_rec_mean"], latent_to_input["input_rec_std"]]
        [latent_to_input["input_rec_mean"]]
    )

    # optimizer
    learning_rate = 0.01
    batch_optimizer = BatchOptimizer(
        whole_dataset_in_device=False,
        batch_size=1,
        max_nb_epochs=100,
        verbose=1,
        optimization_procedure=(updates.rmsprop, {"learning_rate": learning_rate}),
    )

    # Putting all together
    vae = VariationalAutoencoder(input_to_latent_mapping,
                                 latent_to_input_mapping,
                                 batch_optimizer,
                                 rng=rng_mrg.MRG_RandomStreams(seed=state),
                                 input_type=Real,
                                 X_type=T.tensor3,
                                 nb_z_samples=10)
    aa = Autoencoder(input_to_latent_mapping,
                     latent_to_input_mapping,
                     batch_optimizer=batch_optimizer,
                     X_type=T.tensor3)
    # training
    #vae.fit(X[0:1])
    aa.fit(X[0:1])


    # sampling
    #samples = vae.sample(nb=1)
    h,=aa.encode(X[0:1])
    samples,=aa.decode(h)
    svg_content = gen_svg_from_output(samples[0])
    with open("out.svg", "w") as fd:
        fd.write(svg_content)
