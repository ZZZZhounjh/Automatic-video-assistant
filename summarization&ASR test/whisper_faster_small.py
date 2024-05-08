from faster_whisper import WhisperModel
from datetime import datetime
from torch import cuda, bfloat16


start_time = datetime.now()
model_size = "small.en"


model = WhisperModel(model_size, device="cuda", compute_type="int8_float16",device_index=[0])
# or run on GPU with INT8
#model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
#model = WhisperModel(model_size, device="cpu", compute_type="int8")


time_1 = datetime.now()


segments, info = model.transcribe("obama speech_10min.mp3", beam_size=5)

time_2 = datetime.now()

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

with open('speech_text.txt', 'w') as f:
    for segment in segments:
        #print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        
        # Write the text to the file inside the loop
        f.write(segment.text)  # You can add a newline character to separate the texts
        

time_3 = datetime.now()

time_loadmodel = time_1 - start_time
time_transcribe = time_3 - time_1
#time_writefile = time_3 - time_2
time_total = time_3 - start_time

print("load model:" ,time_loadmodel)
print("transcribe:" ,time_transcribe)
#print("write file:" ,time_writefile)
print("total time:" , time_total)