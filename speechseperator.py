# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 12:02:42 2025

@author: burak
"""
import speechbrain as sb
from speechbrain.inference.separation import SepformerSeparation as separator
import torchaudio

import os
from speechbrain.dataio.dataio import read_audio
from IPython.display import Audio, display

import sounddevice as sd





# ham halde kaydı dinlemek için 
signal = read_audio(r"speechbraindeneme\record.wav").squeeze()

_, sr = torchaudio.load(r"speechbraindeneme\record.wav")


os.system(r'start "" speechbraindeneme/record.wav')





# model çalışması
model = separator.from_hparams(source="speechbrain/sepformer-wsj02mix", savedir='pretrained_models/sepformer-wsj02mix')

est_sources = model.separate_file(path=r"record.wav")



for i in range(est_sources.shape[2]):
    torchaudio.save(
        r"speechbraindeneme\record_{i+1}.wav", 
        est_sources[:, :, i].detach().cpu().squeeze().unsqueeze(0), 
        sample_rate=sr
    )

    






# Audio(est_sources[:, :, 0].detach().cpu().squeeze(), rate=8000)
# Audio(est_sources[:, :, 1].detach().cpu().squeeze(), rate=8000)






    


