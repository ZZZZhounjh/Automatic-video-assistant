import whisperx
import gc 

from typing import BinaryIO
import subprocess
import numpy as np
# hard-coded audio hyperparameters
SAMPLE_RATE = 16000

model_size = "small.en"
batch_size = 16 # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy) 
model = whisperx.load_model(model_size, device="cuda", compute_type=compute_type, device_index=[1])

def load_audio(file, sr: int = SAMPLE_RATE):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: BinaryIO
        The audio file object to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    try:
        # Launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI to be installed.
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads",
            "0",
            "-i",
            "pipe:",  # Read from stdin
            "-f",
            "s16le",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sr),
            "-",
        ]
        out = subprocess.run(cmd, input=file.read(), capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


with open("obama speech_10min.mp3", "rb") as f:
    audio = load_audio(f)

result = model.transcribe(audio, batch_size=batch_size)

print(result["segments"]['text']) # before alignment