import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import re
import sys
import os
IS_STOP = False

def state_callback(*args):
    pass

# speech2text = Speech2Text(
#     "model_espnet/exp/asr_train_asr_conformer6_raw_vi_bpe6000_sp/config.yaml",
#     "model_espnet/exp/asr_train_asr_conformer6_raw_vi_bpe6000_sp/44epoch.pth",
#     beam_size=1,
#     ctc_weight=1,
#     penalty=0.4,
#     nbest=1
#     )
chars_to_ignore_regex = '[\,\̀\#\̃\_\̣\=\$\&\̉\?\̀\.\!\́\-\;\:\"\“\%\‘\”\�]'

def remove_special_characters(trans):
    trans = re.sub(chars_to_ignore_regex, '', trans).lower() + ""
    return trans

def start_asr():
    global IS_STOP
    speech, rate = sf.read(sys.argv[1])
    absolute_path = os.path.abspath("pretrain915_finetune_272h_buoc2_thuam")
#     print(absolute_path)
    tokenizer = Wav2Vec2Processor.from_pretrained(absolute_path)
    model = Wav2Vec2ForCTC.from_pretrained(absolute_path)
#     tokenizer = Wav2Vec2Processor.from_pretrained("/home/congpt/GR/data_collection/wav2vec_model/pretrain915_finetune_272h_buoc2_thuam")
#     model = Wav2Vec2ForCTC.from_pretrained("/home/congpt/GR/data_collection/wav2vec_model/pretrain915_finetune_272h_buoc2_thuam")
#     print("asdsadasd")
    input_values = tokenizer(speech, return_tensors="pt", sampling_rate=16000).input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    prediction = tokenizer.batch_decode(predicted_ids)[0]
    text = prediction.lower().replace('[pad]', '')
#     print(text)
    return text

def handler(signum, frame):
    global IS_STOP
    IS_STOP = True

if __name__ == "__main__":
    hypo_text = start_asr()
    print(hypo_text)