## TODO .log/12327 speaker 2, 3, 6, 7, 8, 9, 10 threw errors while processing audio. 2 is not available, 8 had some unicode error and 3, 6, 7, 9, 10 had some intervals missing (i.e. videos missing)
'''
Preprocess audio files
Run it for all speakers
```sh
python data/audio.py -path2data ../dataset/groot/data -path2outdata ../dataset/groot/data -speaker all -preprocess_methods "['log_mel_512']"
```
'''

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pdb
import warnings
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import webrtcvad
from argsUtils import *
from joblib import Parallel, delayed
from tqdm import tqdm

from common import Modality


class Audio(Modality):
  def __init__(self, path2data='../5min_wav_trimmed',
               path2outdata='../preprocessed/5min',
               preprocess_methods=['log_mel_512']):
    super(Audio, self).__init__(path2data=path2data)
    self.path2data = path2data
    self.df = pd.read_csv(Path(self.path2data)/'egocom_cmu_intervals_df.csv', dtype=object)
    self.df.loc[:, 'delta_time'] = self.df['delta_time'].apply(float)
    #self.df.loc[:, 'interval_id'] = self.df['interval_id'].apply(str)
    
    self.path2outdata = path2outdata
    self.interval_id = self.df['interval_id'].apply(str)
    self.speakers = self.df['video_fn'].apply(str) + "_person" + self.df['speaker'].apply(str)
    self.start_time = self.df['start_time'].apply(str)
    self.end_time = self.df['end_time'].apply(str)
    self.preprocess_methods = preprocess_methods
    
  def preprocess(self):
    speakers = self.speakers
    interval_ids = self.interval_id 
    start_time = self.start_time
    end_time = self.end_time
    
    for speaker, interval_id, start_time, end_time in tqdm(zip(list(speakers), list(interval_ids), list(start_time), list(end_time)), desc='speakers', leave=False):
      tqdm.write('Speaker: {}'.format(speaker))
      print('DF SPEAKER', speaker)
      print('INTERVAL IDS',interval_id)

      ## find path to processed files
      ## for each conv_speaker 
      
      # day_1__con_1__part1_person1_0.09_7.64.wav 
      
      parent = Path(self.path2data)/"{}+'_'+{}+'_'+{}+'.wav'".format(speaker, start_time, end_time)
      print('PATH',parent)
      filenames = os.listdir(parent)
      filenames = [filename for filename in filenames]
      print('FILENAMES',filenames)
      filename_dict = {filename.rsplit('_',2)[0]: filename for filename in filenames} # key: conv_person, value: filename 
      print('FILE NAME DICT',filename_dict)

  def save_intervals(self, interval_id, speaker, filename_dict, parent):
    if interval_id in filename_dict:
      ## process data for each preprocess_method
      processed_datas = self.process_interval(interval_id, parent, filename_dict)

      ## save processed_data
      for preprocess_method, processed_data in zip(self.preprocess_methods, processed_datas):
        filename = Path(self.path2outdata)/'processed'/speaker/'{}.h5'.format(interval_id)
        key = self.add_key(self.h5_key, [preprocess_method])
        self.append(filename, key, processed_data)
      return None
    else:
      warnings.warn('interval_id: {} not found.'.format(interval_id))
      return interval_id
  
  def process_interval(self, interval_id, parent, filename_dict):
    ## get filename
    filename = parent/filename_dict[interval_id]

    ## read file
    try:
      y, sr = librosa.load(filename, sr=None, mono=True)
    except:
      return [None] * len(self.preprocess_methods)
    processed_datas = []
    ## process file
    for preprocess_method in self.preprocess_methods:
      processed_datas.append(self.preprocess_map[preprocess_method](y, sr))
    ## return processed output
    return processed_datas

  '''
  PreProcess Methods
  '''
  @property
  def preprocess_map(self):
    return {
      'log_mel_512':self.log_mel_512,
      'log_mel_400':self.log_mel_400,
      'silence':self.silence
      }
      
  def log_mel_512(self, y, sr, eps=1e-10):
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512)
    mask = (spec == 0).astype(np.float)
    spec = mask * eps + (1-mask) * spec
    return np.log(spec).transpose(1,0)

  def log_mel_400(self, y, sr, eps=1e-6):
    y = librosa.core.resample(y, orig_sr=sr, target_sr=16000) ## resampling to 16k Hz
    #pdb.set_trace()
    sr = 16000
    n_fft = 512
    hop_length = 160
    win_length = 400
    S = librosa.core.stft(y=y.reshape((-1)),
                          n_fft=n_fft,
                          hop_length=hop_length,
                          win_length=win_length,
                          center=False)
                                   
    S = np.abs(S)
    spec = librosa.feature.melspectrogram(S=S, 
                                          sr=sr, 
                                          n_fft=n_fft, 
                                          hop_length=hop_length, 
                                          power=1,
                                          n_mels=64,
                                          fmin=125.0,
                                          fmax=7500.0,
                                          norm=None)    
    mask = (spec == 0).astype(np.float)
    spec = mask * eps + (1-mask) * spec
    return np.log(spec).transpose(1,0)

  def silence(self, y, sr, eps=1e-6):
    vad = webrtcvad.Vad(3)
    y = librosa.core.resample(y, orig_sr=sr, target_sr=16000) ## resampling to 16k Hz
    #pdb.set_trace()
    fs_old = 16000
    fs_new = 15
    ranges = np.arange(0, y.shape[0], fs_old/fs_new)
    starts = ranges[0:-1]
    ends = ranges[1:]

    is_speeches = []
    for start, end in zip(starts, ends):
      Ranges = np.arange(start, end, fs_old/100)
      is_speech = []
      for s, e, in zip(Ranges[:-1], Ranges[1:]):
        try:
          is_speech.append(vad.is_speech(y[int(s):int(e)].tobytes(), fs_old))
        except:
          pdb.set_trace()
      is_speeches.append(int(np.array(is_speech, dtype=np.int).mean() <= 0.5))
      is_speeches.append(0)
    return np.array(is_speeches, dtype=np.int)

  @property
  def fs_map(self):
    return {
      'log_mel_512': int(45.6*1000/512), #int(44.1*1000/512) #112 #round(22.5*1000/512)
      'log_mel_400': int(16.52 *1000/160),
      'silence': 15
      }
  
  def fs(self, modality):
    modality = modality.split('/')[-1]
    return self.fs_map[modality]

  @property
  def h5_key(self):
    return 'audio'

def preprocess(args, exp_num):
  path2data = args.path2data #'../dataset/groot/speech2gesture_data/'
  path2outdata = args.path2outdata #'../dataset/groot/data/processed/'
  preprocess_methods = args.preprocess_methods
  audio = Audio(path2data, path2outdata, preprocess_methods)
  audio.preprocess()

if __name__ == '__main__':
  argparseNloop(preprocess)
