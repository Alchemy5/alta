import subprocess
import os
import whisper
import numpy as np
import statistics
import editdistance
from pydub import AudioSegment
from east_detector import EASTDetector
from proc_utils import *
from moviepy.editor import *
from denoiser import pretrained
from denoiser.dsp import convert_audio
import torchaudio
import tkinter as tk
from gingerit.gingerit import GingerIt

from scipy.io import wavfile

class Parser:
    def __init__(self, input_path : str):
        self.video_path = input_path
        assert ".mp4" in self.video_path
        filename = self.video_path[0:self.video_path.index(".mp4")]
        self.audio_path = filename + ".wav"
        if self.audio_path not in os.listdir('.'):
            print("Creating audio file...")          
            command = f"ffmpeg -i {self.video_path} -ab 160k -ac 2 -ar 44100 -vn {self.audio_path}"
            subprocess.call(command, shell=True)
            print("Audio file created.")
        print("Creating Whisper Model...")
        self.whisper = whisper.load_model("small")
        print("Whisper created.")
        self.grammarly = GingerIt()
        self.transcript = {
        0: "oh hello oh you're hearing me yeah I can hear you carry me yeah I'm doing all",
        5: "right wait how are you pretty good I'm pretty good I actually just found out that as a",
        10: "video interview so yeah okay it's okay is this your first time using this",
        17: "platform no I'm just before but from - three months ago so recent here okay",
        22: "I've used I've used other ones but this is my first time using this particular okay yeah okay are you ready software",
        30: "engineer over is looking for a new opportunity I'm looking for a new opportunity yeah yeah",
        37: "what about you I'm working recently in in Bloomberg London office okay yeah",
        43: "well okay yeah I'm in Bay Area California okay yeah okay cool thank you",
        50: "so it's about that I will start asking you first okay so as you see you want some time to",
        58: "read the question first yeah so it's a BST successor search yeah okay",
        74: "okay so in a binary search tree and inorder successor of the node is defined",
        81: "as the node with the smallest key greater than the key of the input node",
        88: "given a node yeah binary search tree you're asked to write a function find",
        94: "inorder successor that returns the inorder successor of input node if it",
        100: "but that has no inorder successor return nor yes so what what you need to make",
        107: "given I know you have to get the smallest node which have a smallest body withers and the given one it exists if",
        113: "it doesn't agree it just return on okay and in the existing code you have the",
        119: "struct for the node which you would code in Java yeah I'm gonna code in Java okay",
        126: "so in you have a class for node that contains the key and lift Android and you have access to the parent itself and",
        133: "the constructor takes a value for the key and for the MS which you need to",
        139: "file which is fine in order successors take same but node and return your node which is a target one and knowledge if",
        145: "it doesn't exist and you have to you have some logic for to insert but you",
        150: "don't need to take care about that okay I'm sorry can you say that last part again you don't need to care about the",
        156: "logic for the insert method okay the insert method yeah it doesn't matter for",
        163: "you okay even not you will not need something like that so all you need just",
        168: "editing right you could here in this part or problem okay",
        182: "and then so the input isn't the necessary of the route it could be any",
        187: "yeah it could be and you know something like the three example on the list okay yeah if I give a new something like",
        193: "knowing okay so I'm searching for the node which contains the element greater",
        199: "than nine and the same time is the smallest one so it will be eleven right okay so am I expecting when I give you",
        205: "nine as an input you got me the note which is eleven this one something the front if I got if I gave you fourteen",
        212: "you need to return the root which is twenty okay",
        218: "because if the first node greater than 14 I see okay so if I do have a nine I",
        228: "want to return eleven yes if I have a twelve I want to return fourteen yes",
        235: "right okay and I do have access to the parent",
        242: "nodes exactly how about to the way up there okay so just yeah going over a",
        247: "couple examples say you give me 1111 yeah then the in order would be twelve",
        255: "right exactly okay",
        266: "so I'm just kind of think of the different cases that we can have here okay",
        285: "so yeah let's just say it use 12 as an example it would be four so so if so if",
        299: "we're using 12 and we're going down the tree it has to be on the right side okay"
        }



    def preprocess(self, path : str):
        """
        Filters and preprocesses audio sample.
        """
        denoising_model = pretrained.dns64().cuda()
        wav, sr = torchaudio.load(path)
        wav = convert_audio(wav.cuda(), sr, denoising_model.sample_rate, denoising_model.chin).cuda()
        wav = wav.cpu()
        torchaudio.save(path, wav, sr)
        # Note: denoising decreases performance significantly as of now

    def parse(self, start_time : int, end_time : int):
        """
        Given start and end time (in seconds), parses video
        and audio files.
        """
        start_time_milli = start_time * 1000
        end_time_milli = end_time * 1000
        new_audio = AudioSegment.from_wav(self.audio_path)
        new_audio = new_audio[start_time_milli:end_time_milli]
        new_audio.export('bit.wav', format="wav")
        audio = whisper.load_audio("bit.wav")
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.whisper.device)
        options = whisper.DecodingOptions()
        audio_str = (whisper.decode(self.whisper, mel, options)).text
        return audio_str
        
    def parse_all(self, output_dir : str):
        """
        Parses whole video file and streams outputs to output
        directory.
        """
        counter = 0
        for start_time in range(0, 291, 25):
            audio_str = self.parse(start_time, start_time + 26)
            with open(f"{output_dir}/{start_time}.txt", "w") as f:
                f.write("EXTRACTED AUDIO TRANSCRIPT:\n")
                f.write(audio_str)
            print(f"Parsed {counter}'th object\n")
            counter += 1

    def parse_all_console(self, save_transcript = False, output_dir = ""):
        """
        Parses whole video file and streams outputs live in console.
        """
        transcript = []
        for start_time in range(0,291,25):  
            audio_str = self.parse(start_time, start_time + 26)
            transcript.append(audio_str)
            print(f"Model Output at {start_time}-{start_time+25}:\n{audio_str}\n")
        if save_transcript:
            with open(f"{output_dir}/transcript.txt", "w") as f:
                f.write("Audio:\n")
                f.write("\n".join(transcript))
        return transcript
            
    def post_process(self, transcript):
        """
        Basically runs model output through a 'Grammarly' to get rid of repetition as well as spelling errors. 
        """
        rough_transcript = " ".join(transcript)
        corrected_transcript = ""
        for i in range(0, len(rough_transcript)-275, 275):
            sub = rough_transcript[i:i+275]
            corrected_transcript += (self.grammarly.parse(sub))['result']
        print(corrected_transcript)
        return corrected_transcript
        # GingerIt isn't much helpful

    def simple_evaluate(self, input : str):
        """
        Simply takes a string and sees how similar it is to YouTube transcript.
        """
        return editdistance.eval(input, " ".join(list(self.transcript.values())))

    def model_evaluate(self):
        """
        Compares model audio output to transcript for model evaluation/comparison purposes.
        """
        all_transcript_text = ""
        all_model_text = ""
        edit_distances = []
        times = list(self.transcript.keys())
        for ind in range(len(times)-1):
            start_time = times[ind]
            end_time = times[ind+1]
            transcript_text = self.transcript[start_time]
            all_transcript_text += (transcript_text + " ")
            model_output = self.parse(start_time, end_time)
            all_model_text += (model_output + " ")
            print(f"Model: {model_output} vs Actual: {transcript_text}\n")
            edit_distance = editdistance.eval(model_output, transcript_text)
            edit_distances.append(edit_distance)
        print(edit_distances)
        print(f"Average edit distance: {statistics.mean(edit_distances)}") # score
        print(f"All transcript text:\n{all_transcript_text}")
        print(f"All model text:\n{all_model_text}")
        print(f"Total edit distance: {editdistance.eval(all_transcript_text, all_model_text)}")
        