from pymagnitude import *
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
import faiss
import re
from operator import itemgetter, attrgetter
import json
import time
import aspell
s = aspell.Speller('lang', 'en')
DIR = 'spelling_corrector/word2vec_api/'
vectors = Magnitude(DIR+"glove.840B.300d.magnitude")

import difflib
def difflib_leven(str1, str2):
    leven_cost = 0
    s = difflib.SequenceMatcher(None, str1, str2)
    for tag, i1, i2, j1, j2 in s.get_opcodes():
       #print('{:7} a[{}: {}] --> b[{}: {}] {} --> {}'.format(tag, i1, i2, j1, j2, str1[i1: i2], str2[j1: j2]))
        if tag == 'replace':
            leven_cost += max(i2-i1, j2-j1)
        elif tag == 'insert':
            leven_cost += (j2-j1)
        elif tag == 'delete':
            leven_cost += (i2-i1)
    return leven_cost

spell_transform_eff = np.load(DIR+'tran_vec.npy')
emb = np.load(DIR+"emb_vec.npy")
id2w = [i.strip() for i in open(DIR+'id2w.txt', 'r').readlines()]
with open(DIR+'w2id.json') as f:
    w2id = json.load(f)

d = emb.shape[1]                            # will be 300 - it's the number of dimensions of each word vector
index = faiss.IndexFlatL2(d)
index = faiss.IndexFlatIP(d)                # This creates the index
index.add(emb)                              # This adds all the word vectors to the index
# print(index.ntotal, 'words now in index')

def getNeighbours(word, transform_vector = spell_transform_eff, c=1.0, neighbours=1000,use_faiss=True):
    
    try:
        word_embeds = np.vstack([emb[w2id[word]]])
    except:
        
        word_embeds = np.vstack([vectors.query(word)])
    
    if use_faiss:
        distances, indices = index.search(
            (word_embeds - transform_vector*c).astype(np.float32), neighbours)
    return indices

def toWords(index_list):
    res = []
    for ind in index_list:
        res.append([id2w[x].lower() for x in ind[:]])
    result = [ x for r in res for x in sorted(set(r),key = r.index) if s.check(x) ] 
    return result

def pick(candidates_list, mis_word , n , topn=1 , use_similarity = False):
    p = []
    not_found = True
    for c in candidates_list:
        #if difflib_leven(c,mis_word)<=3 and len(p)< n:
        if difflib_leven(c,mis_word)<=3 and len(p)< n:
            
            if use_similarity:
                sim = f_s.similarity( mis_word , c )
                p.append((c,difflib_leven(c,mis_word),sim))
            else:
                p.append((c ,difflib_leven(c , mis_word)))
                
           
            not_found = False
            
    if not_found:
        p.append((mis_word,0,0))

        
    if use_similarity:
        p = sorted(p,key=lambda x : (x[1],-x[2]))
    else:
        p = sorted(p , key=(lambda x:x[1]))
        
    #print(p)
    p = [ pp[0] for pp in p ][:topn]
    
    return p

def w2v_candidates(misspell):
    # start = time.time()
    # print(pick(toWords(getNeighbours(mispell,c=1.4)),mispell,50,5))
    # end = time.time()
    # print('花費 :' + str(end - start) + '秒')
    return pick(toWords(getNeighbours(misspell,c=1.4)),misspell,50,5)