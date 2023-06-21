from email.mime import audio
import os
from random import sample
import matplotlib.pyplot as plt 
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchaudio
from pydub import AudioSegment
import soundfile
from IPython.display import Audio, display
import random
import glob
import math
from random import *
import numpy as np
from torchvision import transforms
import librosa
import torchvision

# Dataset class
class audioDataset(Dataset):
    def __init__(self, audio_dir, filepaths, transform, num_samples, sample_rate, device, dataset):
        super().__init__()
        self.device = device
        self.audio_dir = audio_dir
        self.transform = transform
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.dataset = dataset

        if filepaths:
           self.file_paths = filepaths
        else: 
          if (self.dataset == "emotion"):
            self.file_paths = os.listdir(self.audio_dir)
          else:
            self.file_paths = []

        if self.dataset == "gender":
          self.df = pd.read_csv("voxceleb2_age_filtered_2.csv", sep=';')
          self.small_df = self.df
        if self.dataset == "gender":
          if filepaths == None:
            path = self.audio_dir
            for index in self.small_df.index:
              incomplete_path = os.path.join(path,self.small_df['VoxCeleb_ID'][index], self.small_df["video_id"][index])
              dir = os.path.exists(incomplete_path)
              if dir:
                for file_name in os.listdir(incomplete_path):
                  if file_name.endswith(".wav"):
                    self.file_paths.append(os.path.join(incomplete_path, file_name))
              else:
                pass
        
    def __len__(self):
        if (self.dataset == "emotion"):
          return len(os.listdir(self.audio_dir))
        else:
          return len(self.file_paths)  


    def __getitem__(self, index):

        audio_path = os.path.join(self.audio_dir, self.file_paths[index])
        signal, sr = torchaudio.load(audio_path)

        if sr != self.sample_rate:
          resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
          signal = resampler(signal)
        
        signal = self._cut_if_necessary(signal) 
        signal = self._right_pad_if_necessary(signal) 
        signal = self.transform(signal)
        label = self._add_label(self.file_paths[index])

        data_transforms = {
        "train": transforms.Compose([
        transforms.Resize((224,224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])}
        trans = transforms.Lambda(lambda signal: signal.repeat(3, 1, 1) if signal.size(0)==1 else signal)

        signal = trans(signal)
        tr = data_transforms["train"]
        signal = tr(signal)
        return signal, label

    def _cut_signal(self, signal): 
        cut_amount = self.num_samples
        signal = signal[:, int(cut_amount):]
        return signal

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
        
    # labeling
    def _add_label(self, file):
          if (self.dataset == "gender"):
            id = file.split("\\")[6]
            found = self.small_df.loc[self.small_df["video_id"].str.contains(id)]
          if self.dataset == "emotion":
            if "_ANG_" in file or "_angry" in file:
                label = 0
            elif "_DIS_" in file or  "_disgust" in file:
                label = 1
            elif "_FEA_" in file or  "_fear" in file:
                label = 2
            elif "_HAP_" in file or  "_happy" in file:
                label = 3
            elif "_NEU_" in file or  "_neutral" in file:
                label = 4      
            elif "_SAD_" in file or  "_sad" in file:
                label = 5
            return label
    
          if self.dataset == "gender":
            gender = found["gender_wiki"].values[0]
            if "FEMALE" in file or gender == "female":
              label = 0
            elif "MALE" in file or gender == "male":
              label = 1
            return label

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal
    
