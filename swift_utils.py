import subprocess
import os
import whisper
import statistics
import editdistance
from pydub import AudioSegment
from moviepy.editor import *
from denoiser import pretrained
from denoiser.dsp import convert_audio
import torchaudio
import openai
from scipy.io import wavfile
from pathlib import PurePath
import string
import numpy as np
import librosa
import torch
from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC
import matplotlib.pyplot as plt

class Model:
    """
    Class for opensource speech-to-text models.
    """
    def __init__(self, model_name):
        self.model_name = model_name

    def parse(self, input_path : str, start_time = None, end_time = None):
        """
        Given start and end time (in seconds), parses audio files.
        """
        input_path = PurePath(input_path)
        new_audio = AudioSegment.from_file(input_path, input_path.suffix[1:])
        if start_time and end_time:
            start_time_milli = start_time * 1000
            end_time_milli = end_time * 1000
            new_audio = new_audio[start_time_milli:end_time_milli]
        new_audio.export('bit.wav', format="wav")

class Whisper(Model):
    """
    Whisper speech-to-text model.
    """
    def __init__(self, model_name):
        super().__init__(model_name)
        print("Creating Whisper Model...")
        self.whisper = whisper.load_model("medium")
        print("Whisper created.")

    def parse(self, input_path : str, start_time = None, end_time = None):
        """
        Whisper parse function override.
        """
        super().parse(input_path, start_time, end_time)
        audio = whisper.load_audio("bit.wav")
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.whisper.device)
        options = whisper.DecodingOptions()
        audio_str = (whisper.decode(self.whisper, mel, options)).text
        return audio_str

"""
Util functions to import into Swift scripts.
"""
def parseTool(input_path):
    model = Whisper("whisper")
    return model.parse(input_path)