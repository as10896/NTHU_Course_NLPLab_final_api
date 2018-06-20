import onmt
import onmt.io
import onmt.translate
import onmt.ModelConstructor
import io
from collections import namedtuple
from itertools import count
import re
import aspell
s = aspell.Speller('lang', 'en')


def Load_model(model):
    Opt = namedtuple('Opt', ['model', 'data_type', 'reuse_copy_attn', "gpu"])
    opt = Opt(model, "text",False, 0)
    fields, model, model_opt =  onmt.ModelConstructor.load_test_model(opt,{"reuse_copy_attn":False})
    return (fields, model, model_opt)

    

def ch_OpenNMT_candidate(detect_sentence_arr,mes):
    
    fields, model, model_opt = mes[0],mes[1],mes[2]
    ch_candidate = {}
    
    text = '\n'.join(' '.join(word) for word in detect_sentence_arr)
    input_text = io.StringIO(text)
    
    data = onmt.io.build_dataset(fields, "text", input_text, None, use_filter_pred=False)
    data_iter = onmt.io.OrderedIterator(
        dataset=data, device=0,
        batch_size=1, train=False, sort=False,
        sort_within_batch=True, shuffle=False)
    
    # Translator
    scorer = onmt.translate.GNMTGlobalScorer(0,
                                             0,
                                             None,
                                             None)
   
    
    
    translator = onmt.translate.Translator(model, fields,
                                               beam_size=20,
                                               n_best=15,
                                               global_scorer=scorer,
                                               cuda=True)
    builder = onmt.translate.TranslationBuilder(
            data, translator.fields,
            15, False, None)
    
    for batch in data_iter:
        batch_data = translator.translate_batch(batch, data)
        translations = builder.from_batch(batch_data)
        for trans in translations:
            n_best_preds = [" ".join(pred) for pred in trans.pred_sents[:15]]
        
        ch_candidate[' '.join(translations[0].src_raw).replace(' ','')] = n_best_preds
        
        
    
        
        
    
    can = [c.replace(' ','').replace(',','').replace('.','') for ch in ch_candidate.values() for c in ch ]
    can = [c for c in can if s.check(c)]

        
    
    
    return can

