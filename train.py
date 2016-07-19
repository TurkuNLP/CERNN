from keras.models import Sequential, Graph, Model
from keras.layers import Dense, Dropout, Activation, Merge, Input, merge, Flatten

from keras.layers.recurrent import GRU

from keras.callbacks import Callback,ModelCheckpoint
from keras.layers.embeddings import Embedding

import data_generator
import sys
import numpy as np


class CustomCallback(Callback):

    def __init__(self,dev_iter,words):
        self.dev_iter=dev_iter
        self.words=words
        # self.index2label=index2label
        # self.best_mr = 0.0
        # self.dev_labels_text=[]
        # for l in self.dev_labels:
        #     self.dev_labels_text.append(index2label[np.argmax(l)])

    def on_epoch_end(self, epoch, logs={}):

        msdict,tdict=next(self.dev_iter) # fill matrix with next batch
        
        preds=self.model.predict(msdict,verbose=1)
        
        for i in range(0,len(preds)): # show what the network sees
            print("Focus:",self.words[msdict["focus"][i]],file=sys.stdout)
            print("context:",[self.words[t] for t in msdict["context"][i] if t!=0],file=sys.stdout)
            print("predicted:",self.words[np.argmax(preds[i])],file=sys.stdout)
            print("correct:",self.words[np.argmax(tdict[i])],file=sys.stdout)
            print("-"*50,file=sys.stdout)
        
#        preds_text=[]
#        for l in preds:
#            preds_text.append(self.words[np.argmax(l)])
#        correct=[]
#        for l in tdict:
#            correct.append(self.words[np.argmax(l)])
#        print(" ".join(preds_text),file=sys.stdout)
#        print(" ".join(correct),file=sys.stdout)

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





vocabulary_size=100000
batchsize=100
max_sent_len=100
window=10
vec_size=200
gru_width=100

print("test",file=sys.stdout)

# vocabulary
vocab=data_generator.Vocabulary()
vocab.read_vocab("/wrk/jmnybl/pb34_wf_200_v2.bin",vsize=vocabulary_size)
print("Vocabulary size:",len(vocab.words),file=sys.stdout)
print(vocab.words[:10],file=sys.stderr)
sys.stdout.flush()

# matrices
ms=data_generator.Matrices(max_sent_len,len(vocab.words),window,batchsize=batchsize)

#dev_ms=data_generator.Matrices(max_sent_len,len(vocab.words),window,batchsize=10) # another set of matrices for devel data
print("ms created",file=sys.stdout)
sys.stdout.flush()

#Inputs
focus_inp=Input(shape=(1,), name="focus", dtype="int32")
context_inp=Input(shape=(window,), name="context", dtype="int32")


#Word embeddings (initialized with word2vec vectors)
word_emb=Embedding(len(vocab.words), vocab.shape[1], input_length=window, mask_zero=True, weights=[vocab.vectors], trainable=False)

second_emb=Embedding(len(vocab.words), vocab.shape[1], input_length=1, weights=[vocab.vectors], trainable=False)

#Vectors
flattener=Flatten()
focus_vec=flattener(second_emb(focus_inp))


context_vec=word_emb(context_inp)


# Layers: nothing for focus word, gru for context
context_gru=GRU(200)

#focus_dense_out=focus_dense(focus_vec)
context_gru_out=context_gru(context_vec)

# Merge: sum, concat, something else... ???
merged_out=merge([context_gru_out,focus_vec],mode='concat')

# Final dense
final_dense=Dense(200, activation="linear")
final_dense_out=final_dense(merged_out)

# Softmax
softmax_layer=Dense(len(vocab.words), activation="softmax")(final_dense_out)


# compile the model
model=Model(input=[focus_inp,context_inp],output=softmax_layer)

print("ready to compile",file=sys.stdout)
sys.stdout.flush()
model.compile(optimizer="adam",loss="categorical_crossentropy")

print("compiled, training",file=sys.stdout)
sys.stdout.flush()

# save model structure
model_json = model.to_json()
with open("keras_model.json", "w") as json_file:
    json_file.write(model_json)

#import pdb
#pdb.set_trace()
sys.exit()
data_iterator=data_generator.iter_data(sys.stdin)

#dev_iterator=data_generator.iter_data(open("/wrk/jmnybl/fi-ud-dev.conllu"))
#batch_iterd=data_generator.fill_batch(dev_ms,vocab,dev_iterator)

#custom_cb=CustomCallback(batch_iterd,vocab.words)
save_cb=ModelCheckpoint(filepath="keras_model.h5", monitor='loss', verbose=1, save_best_only=False, mode='auto')

#model.fit_generator(data_generator.fill_batch(ms,vocab,data_iterator),10000,1000,callbacks=[custom_cb])

print("fit generator",file=sys.stdout)
sys.stdout.flush()

model.fit_generator(data_generator.fill_batch(ms,vocab,data_iterator),samples_per_epoch=10000,nb_epoch=1000,callbacks=[save_cb],verbose=1)


###-----
###model=Model(input=[src_inp,trg_inp], output=merged_out_flat)
###model.compile(optimizer='adam',loss='mse')

###inf_iter=data_dense.InfiniteDataIterator(src_f_name,trg_f_name,max_iterations=None)
###batch_iter=data_dense.fill_batch(ms,vs,inf_iter)
###import pdb
###pdb.set_trace()
###model.fit_generator(batch_iter,2*len(inf_iter.data),10) 2* because we also have the negative examples

