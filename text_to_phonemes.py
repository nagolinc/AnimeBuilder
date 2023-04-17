from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import load_dataset
import soundfile as sf
import torch

import librosa

import json
from phonecodes import ipa2arpabet

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")

def get_phonemes(audio_file_path):
    # Read audio file
    audio, sample_rate = sf.read(audio_file_path)
    duration=audio.shape[0]/sample_rate
    # Assuming `audio` is a NumPy array containing your audio data
    original_sampling_rate = sample_rate
    target_sampling_rate = 16000
    resampled_audio = librosa.resample(audio, original_sampling_rate, target_sampling_rate)
    model_stride = 0.02  # For Wav2Vec2 models, stride is usually 20ms (0.02s)
    # Tokenize audio
    input_values = processor(resampled_audio, return_tensors="pt", sampling_rate=target_sampling_rate).input_values
    # Retrieve logits
    with torch.no_grad():
        logits = model(input_values).logits
    # Take argmax and decode
    predicted_ids = torch.argmax(logits, dim=-1)
    phoneme_boundaries = []
    prev_phoneme_id = None
    for i, phoneme_id in enumerate(predicted_ids[0]):
        if phoneme_id != prev_phoneme_id:
            phoneme_boundaries.append(i)
            prev_phoneme_id = phoneme_id
    phoneme_durations = [end - start for start, end in zip(phoneme_boundaries[:-1], phoneme_boundaries[1:])]
    start_times = [boundary * model_stride  for boundary in phoneme_boundaries[:-1]]
    stop_times = [(boundary + duration) * model_stride  for boundary, duration in zip(phoneme_boundaries[:-1], phoneme_durations)]
    phoneme_labels = processor.batch_decode(predicted_ids)
    phoneme_timings = []
    for i, (start, stop) in enumerate(zip(start_times, stop_times)):
        if i < len(phoneme_labels):
            p = phoneme_labels[0][i]
        else:
            p = " "
        phoneme_timings.append({
            "start_time": start,
            "stop_time": stop,
            "phoneme": p
        })
    #convert to arpa phonemes
    arpa_timings=[ {'phoneme':ipa2arpabet(x['phoneme'],'en-us'),
    'start_time':x['start_time'],
    'stop_time':x['stop_time']} for x in phoneme_timings]
    return arpa_timings, duration



def parse_phoneme_data(phoneme_data, use_index=True, phindex_location="phindex.json"):
    frame_num = int(phoneme_data[-1]['stop_time'] * 25)
    phone_list = []

    phone_index = 0
    phone_len = len(phoneme_data)
    cur_end = phoneme_data[phone_index]["stop_time"]

    i = 0
    while i < frame_num:
        if i < cur_end * 25:
            phone_list.append(phoneme_data[phone_index]["phoneme"])
            i += 1
        else:
            phone_index += 1
            if phone_index >= phone_len:
                while i < frame_num:
                    phone_list.append(phoneme_data[-1]["phoneme"])
                    i += 1
            else:
                cur_end = phoneme_data[phone_index]["stop_time"]

    with open(phindex_location) as f:
        ph2index = json.load(f)
        
    phoneme_mapping = {
        ' ': 'SIL',
        'o': 'AO',  # Or use 'OW'
        'a': 'AA',  # Or use 'AE' or 'AH'
        'AH0': 'AH',
        'ER0': 'ER',
        'IH0': 'IH',
        'EL': 'L',
        'EM': 'M',
        'EN': 'N',
        'Q': 'SIL',
        'WH': 'W',
        'ː': None,  # Placeholder, will be replaced by the previous phoneme
        'ɐ': 'AH',  # You can use 'AH' as it is a central vowel, similar to ɐ
        'ɾ': 'D',   # You can use 'D' as it is an alveolar sound, similar to ɾ
    }

    if use_index:
        new_phone_list = []
        prev_phoneme = "SIL"
        for p in phone_list:
            if p in phoneme_mapping:
                p = phoneme_mapping[p]
            if p is None:
                p=prev_phoneme            
            if p in ph2index:
                new_phone_list+=[ ph2index[p] ]
            else:
                print("phoneme not found!",p)
                new_phone_list+=[ph2index['SIL']]
            prev_phoneme=p
                
        phone_list=new_phone_list
            

    saves = {"phone_list": phone_list}

    return saves


def processAudio(audio_file_path, use_index=True, phindex_location="phindex.json"):
    phoneme_data, duration = get_phonemes(audio_file_path)
    phonemes = parse_phoneme_data(phoneme_data, use_index=use_index, phindex_location=phindex_location)
    expected_frames = int(duration * 25)
    if len(phonemes['phone_list']) != expected_frames:
        #SIL is repremesnted by 31
        SIL=31
        longer_phonemes=phonemes['phone_list']+([SIL]*(expected_frames-len(phonemes['phone_list'])))
        phonemes['phone_list']=longer_phonemes
    return phonemes


