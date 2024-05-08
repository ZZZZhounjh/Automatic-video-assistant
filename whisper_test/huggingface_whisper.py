import torch
from transformers import pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"

pipe = pipeline(
  "automatic-speech-recognition",
  model="openai/whisper-small.en",
  chunk_length_s=30,
  device=device,
)

audio = "obama speech_10min.mp3"

prediction = pipe(audio, batch_size=8)["text"]
print(prediction)

