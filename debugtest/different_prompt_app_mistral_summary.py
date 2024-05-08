from langchain.llms import LlamaCpp
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import YoutubeLoader, TextLoader
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
import os
import sys
import torch
import tempfile
import transformers
from PIL import Image
from datetime import datetime
from torch import cuda, bfloat16
from transformers import pipeline
import streamlit as st
from streamlit_chat import message
from streamlit_js_eval import streamlit_js_eval

def save_feedback(feedback):
    feedback_dir = "feedback"
    if not os.path.exists(feedback_dir):
        os.makedirs(feedback_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"feedback_{timestamp}.txt"
    filepath = os.path.join(feedback_dir, filename)
    with open(filepath, "w") as file:
        file.write(feedback)

def transcribe(tmp_audio_path):

    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small.en",
    chunk_length_s=30,
    device=device,
    )

    audio = tmp_audio_path
    prediction = pipe(audio, batch_size=8)["text"]

    return prediction.strip()

def summarize(transcript):
    # Laod model
    llm = LlamaCpp(
        streaming = True,
        model_path="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        #model_path="llama-2-7b.Q4_K_M.gguf",
        temperature=0,
        top_p=1, 
        n_ctx=4096,
        verbose=True,
        #n_gpu_layers=1000,
        #n_gpu_layers=-1,#all move to gpu
        #n_batch=256,
        )

    ### Text preprocessing
    target_len = 600
    chunk_size = 3000
    chunk_overlap = 200
    text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    texts = text_splitter.split_text(transcript,)
    docs = [Document(page_content=t) for t in texts[:]]
    
    #LangChain usage


    # summary_general_prompt = PromptTemplate(input_variables=["history", "context", "question"], template="""
    # You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional and informative.
    
    # Context: {context}
    # History: {history}

    # User: {question}
    # Chatbot:"" 
    # """)
    # chatbot_lecture_prompt = PromptTemplate(input_variables=["history", "context", "question"], template="""
    # You are a knowledgeable chatbot, you already have the knowledge of a lecture video transcript. Help with questions of the user 
    #     with use of this lecture video transcript. Your tone should be professional and informative.
    
    # Context: {context}
    # History: {history}

    # User: {question}
    # Chatbot:"" 
    # """)
    # chatbot_speech_prompt = PromptTemplate(input_variables=["history", "context", "question"], template="""
    # You are a knowledgeable chatbot, you already have the knowledge of a speech video transcript. Help with questions of the user 
    #     with use of this speech video transcript. Your tone should be professional and informative.
    
    # Context: {context}
    # History: {history}

    # User: {question}
    # Chatbot:"" 
    # """)
    # chatbot_tutorial_prompt = PromptTemplate(input_variables=["history", "context", "question"], template="""
    # You are a knowledgeable chatbot, you already have the knowledge of a tutorial video transcript. Help with questions of the user 
    #     with use of this tutorial video transcript. Your tone should be professional and informative.
    
    # Context: {context}
    # History: {history}

    # User: {question}
    # Chatbot:"" 
    # """)

    # chatbot_documentary_prompt = PromptTemplate(input_variables=["history", "context", "question"], template="""
    # You are a knowledgeable chatbot, you already have the knowledge of a documentary video transcript. Help with questions of the user 
    #     with use of this documentary video transcript. Your tone should be professional and informative.
    
    # Context: {context}
    # History: {history}

    # User: {question}
    # Chatbot:"" 
    # """)

    # if option == 'Default':
    #     chatbot_prompt = chatbot_general_prompt
    # elif option == 'Lecture':
    #     chatbot_prompt = chatbot_lecture_prompt
    # elif option == 'Speech':
    #     chatbot_prompt = chatbot_speech_prompt
    # elif option == 'Tutorial':
    #     chatbot_prompt = chatbot_tutorial_prompt
    # elif option == 'Documentary':
    #     chatbot_prompt = chatbot_documentary_prompt





    prompt_template = """
                  Do not explain what you are doing. Do not self reference. You are a professional summary writer. 
                  Write a concise summary of the text that cover the key points of the text. and present the results as follows: 
                    - Serveral paragraphs with the following content: Topic, Outline, Description
                    - A key point list in the format of one key point in one paragraph
                    - A markdown table with the definition of the important key terms mentioned
                    ```{text}```
                    SUMMARY:

               """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    refine_template = (
            "Your job is to produce a final summary\n"
            "We have provided an existing summary up to a certain point: {existing_answer}\n"
            "We have the opportunity to refine the existing summary"
            "with some more context below.\n"
            "------------\n"
            "{text}\n"
            "------------\n"
            f"Given the new context, refine the original summary in English within {target_len} words and do not mention the summary is refined."
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
    
    return output_text

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about your video!"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey!"]
        
def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your video transcript", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                output = conversation_chat(user_input, chain, st.session_state['history'])

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

def create_conversational_chain(vector_store, option):
    chatbot_general_prompt = PromptTemplate(input_variables=["history", "context", "question"], template="""
    You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional and informative.
    
    Context: {context}
    History: {history}

    User: {question}
    Chatbot:"" 
    """)
    chatbot_lecture_prompt = PromptTemplate(input_variables=["history", "context", "question"], template="""
    You are a knowledgeable chatbot, you already have the knowledge of a lecture video transcript. Help with questions of the user 
        with use of this lecture video transcript. Your tone should be professional and informative.
    
    Context: {context}
    History: {history}

    User: {question}
    Chatbot:"" 
    """)
    chatbot_speech_prompt = PromptTemplate(input_variables=["history", "context", "question"], template="""
    You are a knowledgeable chatbot, you already have the knowledge of a speech video transcript. Help with questions of the user 
        with use of this speech video transcript. Your tone should be professional and informative.
    
    Context: {context}
    History: {history}

    User: {question}
    Chatbot:"" 
    """)
    chatbot_tutorial_prompt = PromptTemplate(input_variables=["history", "context", "question"], template="""
    You are a knowledgeable chatbot, you already have the knowledge of a tutorial video transcript. Help with questions of the user 
        with use of this tutorial video transcript. Your tone should be professional and informative.
    
    Context: {context}
    History: {history}

    User: {question}
    Chatbot:"" 
    """)

    chatbot_documentary_prompt = PromptTemplate(input_variables=["history", "context", "question"], template="""
    You are a knowledgeable chatbot, you already have the knowledge of a documentary video transcript. Help with questions of the user 
        with use of this documentary video transcript. Your tone should be professional and informative.
    
    Context: {context}
    History: {history}

    User: {question}
    Chatbot:"" 
    """)

    if option == 'Default':
        chatbot_prompt = chatbot_general_prompt
    elif option == 'Lecture':
        chatbot_prompt = chatbot_lecture_prompt
    elif option == 'Speech':
        chatbot_prompt = chatbot_speech_prompt
    elif option == 'Tutorial':
        chatbot_prompt = chatbot_tutorial_prompt
    elif option == 'Documentary':
        chatbot_prompt = chatbot_documentary_prompt

    
    # Create llm
    llm = LlamaCpp(
    streaming = True,
    model_path="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    temperature=0,
    top_p=1, 
    n_ctx=4096,
    verbose=True,
    #GPU
    #n_gpu_layers=1000,
    #n_gpu_layers=-1,#all move to gpu
    #n_batch=256,
    )
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                 retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                 memory=memory, condense_question_prompt=chatbot_prompt)
    return chain

def main():
    # Initialize session state
    initialize_session_state()
    im = Image.open('sricon.png')
    st.set_page_config(page_title=' 🤖Automatic Video Assistant🔗', layout='wide', page_icon = im)

    # Set up the Streamlit app layout
    st.title("🤖 Automatic Video Assistant 🔗")
    st.subheader(" Powered by LangChain + Streamlit")

    hide_default_format = """
        <style>
        #MainMenu {visibility: hidden; }
        footer {visibility: hidden;}
        </style>
        """
    st.markdown(hide_default_format, unsafe_allow_html=True)

    #sidebar
    with st.sidebar:
        st.markdown("# Introduction")
        st.markdown(
        "Automatic Video Assistant is able to summarize videos and answer related questions.")
        st.markdown("You can select specific video types to enhance the assistant's performance.")
        st.markdown("You can input local video or YouTube video link.")
        st.markdown("# Input your video to start!")
        st.markdown("---")
        st.markdown("# Feedback")
        txt = st.text_area(
                "We will continue to improve💪?",
                "Please share your feedback... ",
            )
        if st.button('Submit'):
            save_feedback(txt)
            st.write('Your feedback is submitted!')

        
    option = st.selectbox(
    'Please indicate you video type for better interaction😀',
    ('Default', 'Lecture', 'Speech', 'Tutorial', 'Documentary'))

    st.write('Selected video type:', option)
    

    #User Input File 
    audio_file = st.file_uploader("Upload Video", type=["mp4", "wav","mp3"])

    with st.form('myform', clear_on_submit=True):
        youtube_url = st.text_input("Or enter a YouTube URL")
        submitted = st.form_submit_button('Submit')

    # Check if either YouTube URL or file uploaded
    if (submitted and youtube_url) or audio_file:
        if youtube_url:
            loader = YoutubeLoader.from_youtube_url(youtube_url, add_video_info=False)
        else:
            #transcript = transcribe(audio_file)
            with tempfile.NamedTemporaryFile(delete=False) as tmp_audio_file:
                tmp_audio_file.write(audio_file.read())
                tmp_audio_path = tmp_audio_file.name
            transcript = transcribe(tmp_audio_path)

        with st.expander("See Transcript"):
            if youtube_url:
                transcript = loader.load()
                # Save the transcript to a text file
                with open("transcript.txt", "w", encoding="utf-8") as file:
                    transcript_text = '\n'.join([document.page_content for document in transcript])
                    file.write(transcript_text)
                with open("transcript.txt", "r", encoding="utf-8") as file:
                    transcript = file.read()
            else:
                with open("transcript.txt", "w") as f:
                    f.write(transcript)
            # Display the transcript
            st.write(transcript)
            # Provide a download button for the transcript
            st.download_button("Download Transcript", transcript, key='transcript_download_button')

        st.subheader("Do you want a summary for this video?")
        if 'clicked' not in st.session_state:
            st.session_state.clicked = False
        def click_button():
            st.session_state.clicked = True
        st.button('Generate summary', on_click=click_button)
        #Summarize
        if st.session_state.clicked:
                with st.expander("See Summary", expanded=True):
                        st.header("Summary")
                        summary = summarize(transcript)
                        with open("summary.txt", "w") as f:
                            f.write(summary)
                        with open("summary.txt", "r") as f:
                            for line in f:
                                st.write(line)
                        st.download_button("Download Summary", summary, key='summary_download_button')
    

       
        loader = TextLoader("transcript.txt")
        documents = loader.load()

        st.header("Chatbot🤖")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        text_chunks = text_splitter.split_documents(documents)

        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                                        model_kwargs={'device': 'cuda:1'})
                                        #model_kwargs={'device': 'cpu'}) #Almost the same speed

        # Create vector store
        vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

        # Create the chain object
        chain = create_conversational_chain(vector_store, option)

        display_chat_history(chain)

        if st.button("Click to start with a new video"):
            streamlit_js_eval(js_expressions="parent.window.location.reload()")

if __name__ == "__main__":
    main()
