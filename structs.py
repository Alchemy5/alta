import subprocess
import os
import whisper
import pytesseract
import numpy as np
import cv2
from pydub import AudioSegment
from east_detector import EASTDetector
from proc_utils import *
from moviepy.editor import *

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
        print("Constructing video object...")
        self.clip = VideoFileClip(self.video_path)
        print("Video object constructed")
        print("Creating Whisper Model...")
        self.whisper = whisper.load_model("base")
        print("Whisper created.")
        self.img_proc_detector = EASTDetector()

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

        bit = self.clip.subclip(start_time, start_time+1)
        bit.write_videofile("bit.mp4")
        cam = cv2.VideoCapture("bit.mp4")
        _, frame = cam.read()
        cam.release()
        blank = np.zeros((frame.shape[0], frame.shape[1]))
        norm_img = cv2.normalize(frame, blank, 0, 255, cv2.NORM_MINMAX)
        slices = self.img_proc_detector.get_slices(norm_img)
        binarized = binarize_images(slices, black_on_white=True)
        screen_text_list = [pytesseract.image_to_string(img) for img in binarized]
        screen_text = "\n".join(screen_text_list)
        return (audio_str, screen_text)
        
    def parse_all(self, output_dir : str):
        """
        Parses whole video file and streams outputs to output
        directory.
        """
        counter = 0
        for start_time in range(0, 291, 10):
            audio_str, screen_text = self.parse(start_time, start_time + 10)
            with open(f"{output_dir}/{start_time}.txt", "w") as f:
                f.write("EXTRACTED AUDIO TRANSCRIPT:\n")
                f.write(audio_str + "\n")
                f.write("EXTRACTED SCREEN TEXT:\n")
                f.write(screen_text)
            print(f"Parsed {counter}'th object\n")
            counter = counter + 1
        


