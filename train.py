from keras.models import Sequential, Graph, Model
from keras.layers import Dense, Dropout, Activation, Merge, Input, merge, Flatten

from keras.layers.recurrent import GRU

from keras.callbacks import Callback,ModelCheckpoint
from keras.layers.embeddings import Embedding

import data_generator
import sys


class CustomCallback(Callback):

    def __init__(self, dev_data,dev_labels,index2label,model_name):
        pass
        # self.model_name = model_name
        # self.dev_data=dev_data
        # self.dev_labels=dev_labels
        # self.index2label=index2label
        # self.best_mr = 0.0
        # self.dev_labels_text=[]
        # for l in self.dev_labels:
        #     self.dev_labels_text.append(index2label[np.argmax(l)])

    def on_epoch_end(self, epoch, logs={}):
        pass
        # print logs

        # corr=0
        # tot=0
        # preds = self.model.predict(self.dev_data, verbose=1)
        # preds_text=[]
        # for l in preds:
        #     preds_text.append(self.index2label[np.argmax(l)])

        # print "Micro f-score:", f1_score(self.dev_labels_text,preds_text,average=u"micro")
        # print "Macro f-score:", f1_score(self.dev_labels_text,preds_text,average=u"macro")
        # print "Macro recall:", recall_score(self.dev_labels_text,preds_text,average=u"macro")

        # if self.best_mr < recall_score(self.dev_labels_text,preds_text,average=u"macro"):
        #     self.best_mr = recall_score(self.dev_labels_text,preds_text,average=u"macro")
        #     model.save_weights('./models_gru/' + self.model_name + '_' + str(epoch) + '_MR_' + str(self.best_mr) + '.hdf5')
        #     print 'Saved Weights!'


        # print classification_report(self.dev_labels_text, preds_text)
        # for i in xrange(len(self.dev_labels)):

        # #    next_index = sample(preds[i])
        #     next_index = np.argmax(preds[i])
        #     # print preds[i],next_index,index2label[next_index]

        #     l = self.index2label[next_index]

        #     # print "correct:", index2label[np.argmax(dev_labels[i])], "predicted:",l
        #     if self.index2label[np.argmax(self.dev_labels[i])]==l:
        #         corr+=1
        #     tot+=1
        # print corr,"/",tot





vocabulary_size=10000
batchsize=100
max_sent_len=100
vec_size=200
gru_width=100

# vocabulary
vocab=data_generator.Vocabulary()
vocab.read_vocab("/home/ginter/w2v/pb34_wf_200_v2.bin",vsize=vocabulary_size)
print(len(vocab.words))
print(vocab.words[:10])

# matrices
ms=data_generator.Matrices(max_sent_len,len(vocab.words),batchsize=batchsize)





#Inputs
focus_inp=Input(shape=(1,), name="focus", dtype="int32")
context_inp=Input(shape=(max_sent_len,), name="context", dtype="int32")


#Word embeddings (initialized with word2vec vectors)
word_emb=Embedding(len(vocab.words), vocab.shape[1], input_length=max_sent_len, mask_zero=True, weights=[vocab.vectors])

second_emb=Embedding(len(vocab.words), vocab.shape[1], input_length=1, weights=[vocab.vectors], trainable=False)

#Vectors
flattener=Flatten()
focus_vec=flattener(second_emb(focus_inp))


context_vec=word_emb(context_inp)


# Layers: linear for focus word, gru for context
focus_dense=Dense(200, activation="linear")

context_gru=GRU(200)

focus_dense_out=focus_dense(focus_vec)
context_gru_out=context_gru(context_vec)

# Merge: sum, concat, something else... ???
merged_out=merge([focus_dense_out,context_gru_out],mode='concat')
#merged_out=merge([focus_vec,context_gru_out],mode='concat')

# Softmax
softmax_layer=Dense(len(vocab.words), activation="softmax")(merged_out)


# compile the model
model=Model(input=[focus_inp,context_inp],output=softmax_layer)
model.compile(optimizer="adam",loss="categorical_crossentropy")

#import pdb
#pdb.set_trace()

data_iterator=data_generator.iter_data(sys.stdin)

model.fit_generator(data_generator.fill_batch(ms,vocab,data_iterator),1000,100)


###-----
###model=Model(input=[src_inp,trg_inp], output=merged_out_flat)
###model.compile(optimizer='adam',loss='mse')

###inf_iter=data_dense.InfiniteDataIterator(src_f_name,trg_f_name,max_iterations=None)
###batch_iter=data_dense.fill_batch(ms,vs,inf_iter)
###import pdb
###pdb.set_trace()
###model.fit_generator(batch_iter,2*len(inf_iter.data),10) 2* because we also have the negative examples

