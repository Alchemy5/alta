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

class Dataset:
    """
    Wrapper class for use with popular LibriSpeech dataset.
    """
    def __init__(self, dir_path : str):
        self.dataset_path = dir_path
        transcript_path = ""
        for filename in os.listdir(dir_path):
            if ".trans" in filename:
                transcript_path = f"{dir_path}/{filename}"
                break
        with open(transcript_path, "r") as f:
            transcript_lines = f.readlines()
            self.transcript = {dir_path + "/" + line.split(" ")[0] + ".flac":" ".join(line.split(" ")[1:])[:-1]
                for line in transcript_lines}
        self.filenames = list(self.transcript.keys())

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

class wav2vec(Model):
    """
    wav2vec speech-to-text model.
    """
    def __init__(self, model_name):
        super().__init__(model_name)
        print("Creating wav2vec model...")
        self.tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        print("Wav2vec created.")

    def parse(self, input_path : str, start_time = None, end_time = None):
        """
        wav2vec parse function override.
        """
        super().parse(input_path, start_time, end_time)
        input_audio, _ = librosa.load("bit.wav", sr = 16000)
        input_values = self.tokenizer(input_audio, return_tensors="pt",
            padding="longest").input_values
        with torch.no_grad():
            logits = self.model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
        audio_str = self.tokenizer.batch_decode(predicted_ids)[0]
        return audio_str

class Parser:
    def __init__(self, model_name = "whisper"):
        if model_name == "whisper":
            self.model = Whisper("whisper")
        elif model_name == "wav2vec":
            self.model = wav2vec("wav2vec")
        
        print("Initializing GPT-3...")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        print("GPT-3 initialized.")       

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
        
    def parse_all(self, input_path : str, output_dir : str):
        """
        Parses whole video file and streams outputs to output
        directory.
        """
        counter = 0
        for start_time in range(0, 291, 25):
            audio_str = self.model.parse(input_path, start_time, start_time + 26)
            with open(f"{output_dir}/{start_time}.txt", "w") as f:
                f.write("EXTRACTED AUDIO TRANSCRIPT:\n")
                f.write(audio_str)
            print(f"Parsed {counter}'th object\n")
            counter += 1

    def parse_all_console(self, input_path, save_transcript = False, output_dir = ""):
        """
        Parses whole video file and streams outputs live in console.
        """
        transcript = []
        for start_time in range(0,50,25):  
            audio_str = self.model.parse(input_path, start_time, start_time + 26)
            transcript.append(audio_str)
            print(f"Model Output at {start_time}-{start_time+25}:\n{audio_str}\n")
        if save_transcript:
            with open(f"{output_dir}/transcript.txt", "w") as f:
                f.write("Audio:\n")
                f.write("\n".join(transcript))
        return transcript
            
    def post_process(self, transcript, save_transcript = False, output_dir = ""):
        """
        Basically runs model output through a 'Grammarly' to get rid of repetition as well as spelling errors. 
        """
        rough_transcript = " ".join(transcript)
        corrected_transcript = openai.Completion.create(
            model="text-davinci-003",
            prompt=f"Could you please correct this passage: {rough_transcript}",
            max_tokens = 750,
            temperature=0.1
        )["choices"][0]["text"]
        if save_transcript:
            with open(f"{output_dir}/transcript.txt", "w") as f:
                f.write("Old Audio:\n")
                f.write(rough_transcript + "\n")
                f.write("Corrected Audio:\n")
                f.write(corrected_transcript + "\n")
        return corrected_transcript

    def simple_evaluate(self, input : str):
        """
        Simply takes a string and sees how similar it is to YouTube transcript.
        """
        return editdistance.eval(input, " ".join(list(self.transcript.values())))

    def model_evaluate(self, input_path : str):
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
            model_output = self.model.parse(input_path, start_time, end_time)
            all_model_text += (model_output + " ")
            print(f"Model: {model_output} vs Actual: {transcript_text}\n")
            edit_distance = editdistance.eval(model_output, transcript_text)
            edit_distances.append(edit_distance)
        print(edit_distances)
        print(f"Average edit distance: {statistics.mean(edit_distances)}") # score
        print(f"All transcript text:\n{all_transcript_text}")
        print(f"All model text:\n{all_model_text}")
        print(f"Total edit distance: {editdistance.eval(all_transcript_text, all_model_text)}")
    
    def eval_dataset(self, dataset_path : str, output_path : str):
        """
        Evaluates model on a LibreSpeech dataset.
        """
        dataset = Dataset(dataset_path)
        edit_distances = []
        for filename in dataset.filenames:
            model_output = self.model.parse(filename).upper() # Upper due to LibreSpeech formatting rules.
            model_output_no_punctuation = ""
            for el in model_output:
                if el not in string.punctuation:
                    model_output_no_punctuation += el
            print(f"Model output: {model_output_no_punctuation}")
            edit_distances.append(editdistance.eval(model_output_no_punctuation, 
                dataset.transcript[filename]))
        print(f"Edit distances: {edit_distances}")
        x_axis = np.arange(0.0, len(edit_distances))
        fig, ax = plt.subplots()
        ax.plot(x_axis, edit_distances, 'o')
        ax.set(ylabel='Edit Distance', title = "Dataset Edit Distances")
        ax.grid()
        fig.savefig(output_path)
        
