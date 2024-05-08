import torch
import transformers
from  langchain import LLMChain, HuggingFacePipeline, PromptTemplate
from torch import cuda, bfloat16

from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.llms.openai import OpenAI, OpenAIChat
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from transformers import StoppingCriteria, StoppingCriteriaList

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

### stop token 
stop_list = ['\nHuman:', '\n```\n']

stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
stop_token_ids

stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]
stop_token_ids

# define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False

stopping_criteria = StoppingCriteriaList([StopOnTokens()])





### cutting text
target_len = 500
chunk_size = 2000
chunk_overlap = 500
with open("lecture_text.txt", "r") as f:
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
    max_length=6000, ## Adjuest according to document length
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    stopping_criteria=stopping_criteria,  # without this model rambles during chat
)

llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})


prompt_template = """
              Do not explain what you are doing. Do not self reference. You are an expert academic note taker. Please summarize the text delimited by triple backticks and present the results as follows: 
              - A markdown table with the following columns: Topic, Key Terms, Description (Simplified)
              - A bullet point list with the topic of commonly asked questions about the topic of the text
              - A markdown table with the definition of the important key terms mentioned

              ```{text}```
           """

PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
refine_template = (
        "Your job is to produce a final summary for the lecture text\n"
        "We have provided an existing summary up to a certain point: {existing_answer}\n"
        "We have the opportunity to refine the existing summary"
        "(only if needed) with some more context below.\n"
        "------------\n"
        "{text}\n"
        "------------\n"
        f"Given the new context, refine the original summary in English within {target_len} words and do not mention the summary is refined:"
        "following the previous format"
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
output_file_path = "lecture_summary.txt"
with open(output_file_path, "w") as output_file:
    output_file.write(output_text)
    print(f"Output text saved")
