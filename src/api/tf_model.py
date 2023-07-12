import os

import tensorflow as tf
from numpy import load as npload

async def make_prediction(data, model=None, lookup=None):

    predictions = model.predict(dict(data))

    return await parse_predictions(predictions, lookup)

async def parse_predictions(predictions, lookup):

    predictions = tf.argmax(predictions, axis=-1)
    predictions = lookup(predictions)
    return predictions.numpy()

async def get_architecture(model):

    arch = tf.keras.utils.model_to_dot(model, show_shapes=True, show_layer_names=True, expand_nested=True, rankdir="TD")
    return arch.create_png(prog="dot")
     