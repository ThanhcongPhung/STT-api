from flask import Flask,request
app= Flask(__name__)
import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import re
import sys
import os
import requests
import re
import urllib.request
from flask import jsonify

IS_STOP = False

def state_callback(*args):
    pass

chars_to_ignore_regex = '[\,\̀\#\̃\_\̣\=\$\&\̉\?\̀\.\!\́\-\;\:\"\“\%\‘\”\�]'

def remove_special_characters(trans):
    trans = re.sub(chars_to_ignore_regex, '', trans).lower() + ""
    return trans

def start_asr(url):
    global IS_STOP
    speech, rate = sf.read(url)
    absolute_path = os.path.abspath("wav2vec_model/pretrain915_finetune_272h_buoc2_thuam")
    # print(absolute_path)
    tokenizer = Wav2Vec2Processor.from_pretrained(absolute_path)
    model = Wav2Vec2ForCTC.from_pretrained(absolute_path)
    input_values = tokenizer(speech, return_tensors="pt", sampling_rate=16000).input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    prediction = tokenizer.batch_decode(predicted_ids)[0]
    text = prediction.lower().replace('[pad]', '')
    
    return text

def handler(signum, frame):
    global IS_STOP
    IS_STOP = True
def getFilename_fromCd(cd):
    """
    Get filename from content-disposition
    """
    if not cd:
      return None
    fname = re.findall('filename=(.+)', cd)
    if len(fname) == 0:
      return None
    return fname[0]
@app.route('/v1/transcript')
def index():
  url = request.args.get('url')
  file_name, headers = urllib.request.urlretrieve(url)
  file_type = headers.get('Content-Type')

  if file_type == "audio/wav":
    text = start_asr(file_name)
    return jsonify({'status': 1,
                    'result': {
                      'transcription':text
                    }})

  else:
    return jsonify({'status': 0,
                    'message':"Type must be audio/wav"})
    # text = "Type must be audio/wav"
    # the_response.status=0
    # the_response.message=text
