from .seq2seq_api import ch2ch
from collections import Counter
import re

DIR = 'spelling_corrector/seq2seq_api/'

model= ch2ch.Load_model(DIR+'gec_model/baseline/demo-ch_acc_97.35_ppl_1.12_e13.pt')
model= ch2ch.Load_model(DIR+'gec_model/baseline-mergedata/demo-ch_acc_92.47_ppl_1.67_e13.pt')

def s2s_candidates(misspell):
    return ch2ch.ch_OpenNMT_candidate([misspell],model)[:5]
