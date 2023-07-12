import tensorflow as tf
from numpy import load as npload

async def make_prediction(data):
    model = tf.keras.models.load_model("/app/model/prod_model")
    predictions = model.predict(data)

    return await parse_predictions(predictions)

async def parse_predictions(predictions):
    vocab = npload.load("/app/model/vocab_prod_model.npy")
    lookup = tf.keras.layers.StringLookup(vocabulary=vocab, invert=True)

    predictions = tf.argmax(predictions, axis=-1)
    predictions = lookup(predictions)
    return predictions.numpy().tolist()
     