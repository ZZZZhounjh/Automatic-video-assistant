import torch
import transformers
from  langchain import LLMChain, HuggingFacePipeline, PromptTemplate
from torch import cuda, bfloat16

from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.llms.openai import OpenAI, OpenAIChat
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter


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

### cutting text
target_len = 500
chunk_size = 2000
chunk_overlap = 500
with open("meeting_text.txt", "r") as f:
        raw_text = f.read()
# Split the source text
text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
texts = text_splitter.split_text(
        raw_text,
    )

    # Create Document objects for the texts
docs = [Document(page_content=t) for t in texts[:]]


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


prompt_template = """
              Act as a professional meeting minutes writer. Write a summary of the following text delimited by triple backticks.
              Return your response which covers the key points of the text.
              Format: Meeting summary
              Tasks:
              - highlight action items and owners
              - highlight the agreements
              - Use bullet points if needed
              following the format:
              Participants: <participants>
              Discussed: <Discussed-items>
              Agreement:<Agreements>
              Follow-up actions: <a-list-of-follow-up-actions-with-owner-names>
              ```{text}```
              SUMMARY:
           """
PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
refine_template = (
        "Your job is to produce a final summary for the meeting\n"
        "We have provided an existing summary up to a certain point: {existing_answer}\n"
        "We have the opportunity to refine the existing summary"
        "(only if needed) with some more context below.\n"
        "------------\n"
        "{text}\n"
        "------------\n"
        f"Given the new context, refine the original summary in English within {target_len} words and do not mention the summary is refined:"
        "following the format"
        "Participants: <participants>"
        "Discussed: <Discussed-items>"
        "Agreement:<Agreements>"
        "Follow-up actions: <a-list-of-follow-up-actions-with-owner-names>"
    )
refine_prompt = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template,
    )
chain = load_summarize_chain(
        llm,
        chain_type="refine",
        return_intermediate_steps=True,
        question_prompt=PROMPT,
        refine_prompt=refine_prompt,
    )


resp = chain(docs)
output_text = resp["output_text"]

print("Output Text:")
print(output_text)

### save output text
output_file_path = "meeting_summary_7b.txt"
with open(output_file_path, "w") as output_file:
    output_file.write(output_text)

print(f"Output text saved")