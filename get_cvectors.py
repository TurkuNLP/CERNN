from keras.models import Sequential, Graph, Model, model_from_json
from keras.layers import Dense, Dropout, Activation, Merge, Input, merge, Flatten

from keras.layers.recurrent import GRU

from keras.callbacks import Callback,ModelCheckpoint
from keras.layers.embeddings import Embedding

import data_generator
import sys
import numpy as np


# load model
with open('keras_model.json', 'r') as f:
    model = model_from_json(f.read())
# load weights
model.load_weights("keras_lm.model")
print("Model loaded")

# freeze layers, is this needed?

# compile
model.compile(optimizer="adam",loss="categorical_crossentropy")

my_layer=model.get_layer("dense_1")

print(my_layer)


from keras import backend as K
def get_activations(model, layer_name, X):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.get_layer(layer_name).output,])
    activations = get_activations([X,0])
    return activations

### create sample input here ###

vocabulary_size=100000
batchsize=100
max_sent_len=100
window=10
vec_size=200
gru_width=100

# read vocabulary
vocab=data_generator.Vocabulary()
vocab.read_vocab("/wrk/jmnybl/pb34_wf_200_v2.bin",vsize=vocabulary_size)

ms=data_generator.Matrices(max_sent_len,len(vocab.words),window,batchsize=1)

ms.focus[0]=vocab.get_id("kuusi")
context="pitäisi kaataa metsästä tuo iso , saisi polttopuuta".split(" ")
for i in range(len(context)):
    ms.context[0,i]=vocab.get_id(context[i])

activations=get_activations(model, "dense_1", ms)
print(activations)
print(type(activations))
