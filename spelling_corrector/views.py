from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse, JsonResponse

from .conventional import sentence_correction
from .word2vec import w2v_candidates
from .seq2seq import s2s_candidates

import json

# Create your views here.

@csrf_exempt
def conventional_spelling_corrector(request):
    if request.method == "POST":
        # if request.POST:
            # json_data = request.POST
        if request.body:
            json_data = json.loads(request.body)
            sentences = json_data["sentence"]
            # print(sentences.encode('utf-8'))
            sentence = sentences.strip().split('\n')[-1]
            # print(sentence)
            response = sentence_correction(sentence)
            return JsonResponse(response, safe=False)
        else:
            return HttpResponse("Your request body is empty.")
    else:
        return HttpResponse("Please use POST.")
    
@csrf_exempt
def word2vec(request):
    if request.method == "POST":
        if request.body:
            json_data = json.loads(request.body)
            misspell = json_data["misspell"]
            response = w2v_candidates(misspell)
            return JsonResponse(response, safe=False)
        else:
            return HttpResponse("Your request body is empty.")
    else:
        return HttpResponse("Please use POST.")

@csrf_exempt
def seq2seq(request):
    if request.method == "POST":
        if request.body:
            json_data = json.loads(request.body)
            misspell = json_data["misspell"]
            response = s2s_candidates(misspell)
            return JsonResponse(response, safe=False)
        else:
            return HttpResponse("Your request body is empty.")
    else:
        return HttpResponse("Please use POST.")