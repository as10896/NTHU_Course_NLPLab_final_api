from functools import reduce
from math import log
from pprint import pprint
from collections import Counter, defaultdict, OrderedDict
import re
import difflib

from .LinggleAPI import *
from .NetSpeakAPI import *
from .word2vec import w2v_candidates
from .seq2seq import s2s_candidates


import aspell
s = aspell.Speller('lang', 'en')

def find_ngram_position(full, part):
    full = full.split()
    part = part.split()
    length = len(part)
    for i, sublist in enumerate((full[i:i+length] for i in range(len(full)))):
        if part == sublist:
            return [i+c for c in range(length)]
    return []

cache = {}
def CallLinggle(text, N=3):
    SE = Linggle()
    # SE = NetSpeak()
    if (text, N) in cache:
        return cache[text, N]  
    test = text.split()
    d = {}   
    for i in range(len(test) - (N-1)):
        res = SE.search(' '.join(test[i:i + N]))
        if len(res) == 0:
            d[' '.join(test[i:i + N])] = 0
        else:
            d[res[0][0]]=res[0][1]
    d_output = {}
    for trigram, count in d.items():
        if count == 0:
            d_output[trigram] = 1
        else:
            d_output[trigram] = count
    cache[text, N] = d_output
    return cache[text, N]

def sentence_differ(w, c):
    d = difflib.Differ()
    w, c = w.split(), c.split()
    
    output = []
    for s in list(d.compare(w, c)):
        if s[0] == " " or s[0] == "?": continue
        elif s[0] == "-":
            wrong = s[2:]
        elif s[0] == "+":
            correct = s[2:]
            output.append((wrong, correct))
    return output

def sentence_word_diff(w, c):
    w = list(enumerate(w.split()))
    c = list(enumerate(c.split()))
    output_sentence = []
    for i in range(len(w)):
        if w[i][1] == c[i][1]:
            output_sentence.append(c[i][1])
        else:
            output_sentence.append('[-%s-] {+%s+}'%(w[i][1], c[i][1]))
    return " ".join(output_sentence)

def detect_nonword_error(text, N=3):
    nonword_error = []
    for position, word in enumerate(text.split()):
        if not s.check(word):
            nonword_error.append((position, word))

    error_candidates = {}
    for position, word in nonword_error:
        error_candidates[position, word] = list(OrderedDict.fromkeys([candidate.lower() for candidate in s.suggest(word) if len(candidate.split())==1 and len(candidate.split('-'))==1]))[:5]
        
    text_copy = text.split()
    for (position, word), candidates in error_candidates.items():
        text_copy[position] = candidates[0]
    sentence_edits = [" ".join(text_copy)]
    
    for (position, word), candidates in error_candidates.items():
        sentence_edits_copy = sentence_edits[:]
        for sentence in sentence_edits_copy:
            sentence = sentence.split()
            for candidate in candidates[1:]:
                sentence[position] = candidate
                sentence_edits.append(" ".join(sentence))
    
    d = {}
    for sentence_edit in sentence_edits:
        d_sub = CallLinggle(sentence_edit, N)
        d[sentence_edit] = reduce(lambda x, y: x*y, d_sub.values())
        print("%s:\t%d"%(sentence_edit, d[sentence_edit]))
        
    d = sorted(d.items(), key=lambda x: x[1], reverse=True)
    return d[0][0]

def sentence_correction(text, N=3):
    isupper = True
    if not text.split()[0][0].isupper(): isupper = False

    if len(text.split()) == 1: 
        candidates = list(OrderedDict.fromkeys([candidate.lower() for candidate in s.suggest(text) if len(candidate.split())==1]))
        details = []
        answer = candidates[0]
        if text != answer:
            details.append({'error': text, 'candidates': candidates[:5],'seq2seq': s2s_candidates(text), 'w2v': w2v_candidates(text), 'correction': answer})
            word_diff = '[-%s-] {+%s+}' % (text, answer)
            return {'before': text, 'after': answer, 'details': details, 'word_diff': word_diff}
        return {'before': text, 'after': text, "details": details, 'word_diff': text}
    
    if len(text.split()) == 2: N=2
        
    text = text.lower()
    original_text = text
    nonword_correction_text = detect_nonword_error(text, N)
    sentence_count = CallLinggle(nonword_correction_text, N)
    minimum = min(sentence_count.values())
    possible_error_trigrams = [list(zip(k.split(), find_ngram_position(nonword_correction_text, k))) for k,v in sentence_count.items() if v == minimum or v < 500]
    error_words = list(OrderedDict.fromkeys([possible_error for possible_error_trigram in possible_error_trigrams for possible_error in possible_error_trigram]))
    
    sentence_edits = [nonword_correction_text]
    for error_word, error_position in error_words:
        candidates = list(OrderedDict.fromkeys([candidate.lower() for candidate in s.suggest(error_word) if len(candidate.split())==1 and len(candidate.split('-'))==1]))[:5]
        sentence_edits_copy = sentence_edits[:] ## use pass by value, or it will cause infinite loop
        for sentence in sentence_edits_copy:
            sentence = sentence.split()
            for candidate in candidates:
                sentence[error_position] = candidate
                sentence_edits.append(' '.join(sentence))
                
    d = {}
    for sentence_edit in sentence_edits:
        d_sub = CallLinggle(sentence_edit, N)
        d[sentence_edit] = reduce(lambda x, y: x*y, d_sub.values())
        print("%s:\t%d"%(sentence_edit, d[sentence_edit]))
        
    correction_sentence = sorted(d.items(), key=lambda x:x[1], reverse=True)[0][0]

    details = []
    
    for error, correct in sentence_differ(original_text, correction_sentence):
        candidates = list(OrderedDict.fromkeys([candidate.lower() for candidate in s.suggest(error) if len(candidate.split())==1 and len(candidate.split('-'))==1]))[:5]
        details.append({'error': error, 'candidates': candidates, 'seq2seq': s2s_candidates(error), 'w2v': w2v_candidates(error), 'correction': correct})
    word_diff = sentence_word_diff(original_text, correction_sentence)
    if isupper:
        original_text = original_text.split()
        original_text[0] = original_text[0].capitalize()
        original_text = " ".join(original_text)
        correction_sentence = correction_sentence.split()
        correction_sentence[0] = correction_sentence[0].capitalize()
        correction_sentence = " ".join(correction_sentence)
        word_diff = word_diff.split()
        word_diff[0] = word_diff[0].capitalize()
        word_diff = " ".join(word_diff)
    return {'before': original_text, 'after': correction_sentence, 'details': details, 'word_diff': word_diff}