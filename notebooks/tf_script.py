from ds.data.sets import load_sets_v2
import joblib
import pandas as pd
import tensorflow as tf
import numpy as np
from pathlib import Path

# set vars
BATCH_SIZE = 10000

# load data
datasets = joblib.load("../data/processed/datasetsjmone")

# drop unused columns
drops = [
    "review_overall",
    "beer_name",
]
for data in datasets:
    if isinstance(data, pd.DataFrame):
        for column in drops:
            if column in data.columns:
                data.drop(columns=column, inplace=True)

# split data
x_train, y_train, x_test, y_test, x_validation, y_validation = datasets

# fill na with missing value
x_train = x_train.fillna(-9)
x_test = x_test.fillna(-9)

# convert y to tesnor with tf type
y_vocab = sorted(set(y_train))
y_tensor = tf.convert_to_tensor(y_train, dtype=tf.string)

# create lookup layer and encode target onehot
target_lookup = tf.keras.layers.StringLookup(vocabulary=y_vocab, output_mode='one_hot')
target = target_lookup(y_tensor)

# create training dataset
train = tf.data.Dataset.from_tensor_slices((dict(x_test), target))
train = train.batch(BATCH_SIZE)

# create tes dataset
y_test_tensor = tf.convert_to_tensor(y_test)
y_test_target = target_lookup(y_test_tensor)
test = tf.data.Dataset.from_tensor_slices((dict(x_test), y_test_target))
test = test.batch(BATCH_SIZE)

# calculate biases
y, idx, count = tf.unique_with_counts(y_tensor)
count = tf.cast(count, tf.float32)
total_tkns = tf.math.reduce_sum(count)
tkn_probs = count/total_tkns
log_probs = tf.math.log(tkn_probs)
vocab_size = target_lookup.vocabulary_size()
vocab = tf.convert_to_tensor(target_lookup.get_vocabulary())

# create subsitution table
init_sub = tf.lookup.KeyValueTensorInitializer(y,log_probs)
couttbl = tf.lookup.StaticHashTable(init_sub, default_value=-1e9)
bias_vector = tf.zeros(vocab_size,)
bias_vector = couttbl.lookup(vocab)
# annoying conversion because of saving issues
bias_vector = bias_vector.numpy()
# create initilizer
output_bias_init = tf.keras.initializers.Constant(bias_vector)

# setup features
numeric_feature_names = [
    "review_aroma",
    "review_appearance",
    "review_palate",
    "review_taste",
    "beer_abv"
]
numeric_features = x_test.loc[:,numeric_feature_names]

categorical_feature_names = [
    "brewery_name"
]

# from tf docs to create inputs
def stack_dict(inputs, fun=tf.stack):
    values = []
    for key in sorted(inputs.keys()):
      values.append(tf.cast(inputs[key], tf.float32))

    return fun(values, axis=-1)

inputs = {}
preprocessed = []

# create input for all columns
for name, column in x_train.items():
  if type(column[0]) == str:
    dtype = tf.string
  else:
    dtype = tf.float32

  inputs[name] = tf.keras.Input(shape=(), name=name, dtype=dtype)

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(stack_dict(dict(numeric_features)))

# shape (None, ) per input
numeric_inputs = {}
for name in numeric_feature_names:
  numeric_inputs[name]=inputs[name]

numeric_inputs = stack_dict(numeric_inputs)
numeric_normalized = normalizer(numeric_inputs)

# ends as shape (None, 5)
preprocessed.append(numeric_normalized)

# categorical features
for name in categorical_feature_names:
  vocab = sorted(set(x_train[name]))

  if type(vocab[0]) is str:
    lookup = tf.keras.layers.StringLookup(vocabulary=vocab, output_mode='one_hot')
  else:
    lookup = tf.keras.layers.IntegerLookup(vocabulary=vocab, output_mode='one_hot')

  x = inputs[name][:, tf.newaxis]
  x = lookup(x)
  preprocessed.append(x)

# combine
preprocesssed_result = tf.concat(preprocessed, axis=-1)
preprocessor = tf.keras.Model(inputs, preprocesssed_result)

proc = preprocessor(inputs)

# model body
body = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(2048, activation='relu'),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(2048, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(target_lookup.vocabulary_size(), activation='softmax', bias_initializer=output_bias_init, name="output_layer")
    ]
)

# compile model
result = body(proc)
model = tf.keras.Model(inputs, result)

model.compile(optimizer='adam',
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                metrics=['accuracy'])

# setup for trian
earlystop_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0.000001,
    patience=32,
    verbose=0,
    restore_best_weights=True,
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='loss',
    factor=0.1,
    patience=16,
    verbose=0,
    mode='auto',
    min_delta=0.0001,
    cooldown=4,
    min_lr=0.00000000001
)

history = model.fit(
    train,
    epochs=20,
    batch_size=BATCH_SIZE,
    validation_data=test,
    callbacks=[        
        earlystop_callback,
        reduce_lr
    ]
)

# saving
def get_version_num(path):
    return int(path.stem.split("_")[-1])

base_path = Path("../models/gpu/")

model_name = "goodmod"
existing_versions = base_path.glob(f"{model_name}*")
existing_versions = [get_version_num(model_path) for model_path in existing_versions]
if len(existing_versions) == 0:
    version_num = 1
else:
    version_num = max(existing_versions) + 1

this_model = f"{model_name}_{version_num}"

save_path = base_path.joinpath(this_model)

target_lookup_path = base_path.joinpath(f"vocab_{this_model}")

target_vocab = target_lookup.get_vocabulary()
target_vocab = np.array(target_vocab)


np.save(target_lookup_path, target_vocab)
model.save(save_path, save_format="keras")
