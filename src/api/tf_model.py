import os

import tensorflow as tf
from numpy import load as npload

async def make_prediction(data):
    if os.environ["CURRENT_CONTAINER"] == "api_dev":
        model = tf.keras.models.load_model("/home/projects/dev/app/model/prod_model")
    else:
        model = tf.keras.models.load_model("/app/model/prod_model")

    predictions = model.predict(dict(data))

    return await parse_predictions(predictions)

async def parse_predictions(predictions):
    if os.environ["CURRENT_CONTAINER"] == "api_dev":
        vocab = npload("/home/projects/dev/app/model/vocab_prod_model.npy")
    else:
        vocab = npload("/app/model/vocab_prod_model.npy")

    lookup = tf.keras.layers.StringLookup(vocabulary=vocab, invert=True)

    predictions = tf.argmax(predictions, axis=-1)
    predictions = lookup(predictions)
    return predictions.numpy()

async def get_architecture():
    if os.environ["CURRENT_CONTAINER"] == "api_dev":
        model = tf.keras.models.load_model("/home/projects/dev/app/model/prod_model")
    else:
        model = tf.keras.models.load_model("/app/model/prod_model")

    arch = tf.keras.utils.plot_model(model, to_file="./models/fancy_model_pic.png", show_shapes=True, show_layer_names=True, expand_nested=True, rankdir="TD")
    return arch
     