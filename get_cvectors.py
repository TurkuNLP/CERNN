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
    trained_model = model_from_json(f.read())
    trained_model.load_weights("keras_model.h5")
    trained_model.compile(optimizer="adam",loss="categorical_crossentropy")


vocabulary_size=100000
batchsize=100
max_sent_len=100
window=10
vec_size=200
gru_width=100

# read vocabulary
vocab=data_generator.Vocabulary()
vocab.read_vocab("/home/ginter/w2v/pb34_wf_200_v2.bin",vsize=vocabulary_size)

# create truncated model and load weights from trained model
#Inputs
focus_inp=Input(shape=(1,), name="focus", dtype="int32")
context_inp=Input(shape=(window,), name="context", dtype="int32")

#Word embeddings (initialized with word2vec vectors)
word_emb=Embedding(len(vocab.words), vocab.shape[1], input_length=window, mask_zero=True, weights=trained_model.get_layer('embedding_1').get_weights(), trainable=False) #embedding_1
second_emb=Embedding(len(vocab.words), vocab.shape[1], input_length=1, weights=trained_model.get_layer('embedding_2').get_weights(), trainable=False) #embedding_2

#Vectors
flattener=Flatten()
focus_vec=flattener(second_emb(focus_inp))
context_vec=word_emb(context_inp)

# Layers: nothing for focus word, gru for context
context_gru=GRU(200,weights=trained_model.get_layer('gru_1').get_weights(),trainable=False)
context_gru_out=context_gru(context_vec)

# Merge
merged_out=merge([context_gru_out,focus_vec],mode='concat')

# Final dense
final_dense=Dense(200, activation="linear",weights=trained_model.get_layer('dense_1').get_weights(),trainable=False)
final_dense_out=final_dense(merged_out)

# ...stop here
# (maybe trained_model.pop() would have done the same)


# compile the model
model=Model(input=[focus_inp,context_inp],output=final_dense_out)
model.compile(optimizer="adam",loss="categorical_crossentropy")



### create sample input here ###

vectors=[]

# read data from file
f=open("test_data.txt","rt",encoding="utf-8")
ms=data_generator.Matrices(max_sent_len,len(vocab.words),window,batchsize=1)

for line in f:
    line=line.strip()
    if not line:
        continue
    focus,context=line.split("\t",1)
    print(focus,context)
    context=context.split(" ")
    ms.clean()
    ms.focus[0]=vocab.get_id(focus)
    for i,cword in enumerate(context):
        ms.context[0,i]=vocab.get_id(cword)
    # predict
    activations=model.predict({"context":ms.context,"focus":ms.focus})[0]
    activations/=np.linalg.norm(x=activations,ord=None)
    vectors.append(activations)
    if len(vectors)>1:
        print(np.dot(vectors[-2],vectors[-1]))
    


