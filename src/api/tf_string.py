from numpy import load as npload
from tensorflow.keras.layers import StringLookup

async def tf_string_convert(string, invert=False):
    vocab = npload("/app/model/vocab_prod_model.npy")
    if invert is False:    
        lyr = StringLookup(vocabulary=vocab, output_mode='one_hot')
    else:
        lyr = StringLookup(vocabulary=vocab, invert=True)

    return lyr(string)