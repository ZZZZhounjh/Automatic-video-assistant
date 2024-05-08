import torch
import transformers
from  langchain import LLMChain, HuggingFacePipeline, PromptTemplate
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


segments, info = model.transcribe("Obama speech.mp3", beam_size=5)

time_2 = datetime.now()

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

with open('whisper_text.txt', 'w') as f:
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





model_id = 'meta-llama/Llama-2-7b-chat-hf'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

# begin initializing HF items, you need an access token
hf_auth = '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'
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    use_auth_token=hf_auth
)

# enable evaluation mode to allow model inference
model.eval()

print(f"Model loaded on {device}")

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

### model loading done
### Creating the Summarization Pipeline

### parameters setting
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    max_length=3000, ## Adjuest according to document length
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})

template = """
              Write a summary of the following text delimited by triple backticks.
              Return your response which covers the key points of the text.
              ```{text}```
              SUMMARY:
           """
           
### Elevating Summarization with Langchain
prompt = PromptTemplate(template=template, input_variables=["text"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

# Read text from the file
file_path = "whisper_text.txt"
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

# Run Langchain and save the summary to a file

summary = llm_chain.run(text)

print("Output Text:")
print(summary)

summary_file_path = "text_summary.txt"  
with open(summary_file_path, 'w', encoding='utf-8') as summary_file:
    summary_file.write(summary)

print(f"Summary saved")