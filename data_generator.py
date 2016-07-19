import lwvlib
import conllutil3 as cu
import numpy as np
import sys

class Vocabulary(object):

    def __init__(self):
        pass

    def read_vocab(self,fname,vsize=10000):
        """ read vocabulary from wvlib model """
        model=lwvlib.load(fname,vsize,vsize)
        

        self.words=model.words
        self.words.insert(0,"<MASK>")
        self.words.insert(1,"<UNK>")
        self.shape=(len(self.words),model.vectors.shape[1])

        self.word_to_dim=dict((w,i) for i,w in enumerate(self.words))

        self.vectors=np.zeros((vsize+2,model.vectors.shape[1]),np.float)
        
        for i,row in enumerate(model.vectors):
            self.vectors[i+2]=row
        self.vectors[1]=self.vectors[np.random.randint(2,len(self.words))] # take a random vector for unk # TODO: average of something...


    def get_id(self,word):
        if word not in self.words: # oov
            return self.word_to_dim["<UNK>"]
        else:
            return self.word_to_dim[word]

class Matrices(object):

    def __init__(self,max_sent_len,vocablen,window,batchsize=100):
        self.batchsize=batchsize
        self.max_sent_len=max_sent_len
        self.window=window
        self.context=np.zeros((batchsize,window),np.int)
        self.focus=np.zeros((batchsize,1),np.int)
        self.target=np.zeros((batchsize,vocablen),np.int)
        self.mdict={"context":self.context,"focus":self.focus,"target":self.target}

    def clean(self):
        for key,m in self.mdict.items():
            m.fill(0)
       

#max_sent_len=100

def iter_data(f):
    counter=0
    for comm,sent in cu.read_conllu(f):
        if len(sent)==1:
            continue
        words=[t[cu.FORM] for t in sent]
        yield words
        counter+=1
        if counter%10000==0:
            print(counter,file=sys.stderr)

def fill_batch(ms,vocab,iterator):

    row=0
    for words in iterator:
        if len(words)>ms.max_sent_len:
            continue
        for i in range(0,len(words)): # i is focus
            if vocab.get_id((words[i]))==vocab.get_id("<UNK>"): # skip if focus is unknown...
                continue
            for j in range(max(0,i-int(ms.window/2)),min(len(words),i+int(ms.window/2))): # j is target, rest is context
                if i==j or vocab.get_id((words[j]))==vocab.get_id("<UNK>"): # skip if target is unknown...
                    continue
                ms.focus[row]=vocab.get_id(words[i])
                ms.target[row,vocab.get_id(words[j])]=1
                column=0
#                for z,word in enumerate(words):
                for z in range(max(0,i-int(ms.window/2)),min(len(words),i+int(ms.window/2))):
                    if z==i or z==j:
                        continue
                    word=words[z]
                    ms.context[row,column]=vocab.get_id(word)
                    column+=1
                row+=1
                if row==ms.batchsize:
                    yield ms.mdict,ms.target
                    row=0
                    ms.clean()
    # TODO: you will lose the last, unfinished batch...
    # return dictionary of those matrices (...or tuple), Inputs can then refer to these names

if __name__=="__main__":


    vocab=Vocabulary()
    vocab.read_vocab("/home/ginter/w2v/pb34_wf_200_v2.bin",vsize=10000)
    print(len(vocab.words))
    print(vocab.words[:10])

   # print(vocab.model.vectors.shape)
    print(vocab.vectors.shape)

    ms=Matrices(max_sent_len,len(vocab.words))
#    print(ms)

#    for b,u in fill_batch(ms,vocab,iter_data(sys.stdin)):
#        for i in range(0,ms.batchsize):
#            print("focus:",vocab.words[b["focus"][i]],sep=" ")
#            print("target:",vocab.words[b["target"][i]],sep=" ")
#            print("context:",u" ".join(vocab.words[c] for c in b["context"][i]))
    



