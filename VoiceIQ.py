

'''
Description;

The aim of this implementation is developing an algorithm that by converting audience audio to text format online and injects it into an LLM, 
making a good environment to aid it with their questions.  

This simple code is the first step in developing an "Assistant voice" to help you in your daily work, building automation, etc.
The code is developing to execute in termainal and to use it in graphical space you can use "QTPY" library.
The implementation process is as follows;
1- Summoning necessery libraries 2- Giving access to Mic 3- Circulation part (Turning mic on (It should be opened and then turned off) - 
Denoising Mic voice - Starting recording process - Sending recorded file to Google - 
Getting text format of voice - injecting extracted text to LLM - Showing consequences)
Some errors you may face in this project are inaccessibility to the internet and the microphone. To deal with them, we use the "try-except" method.

Notice: To execute this code, I used an 840M GeForce graphics card

Versions and libraries;

speechrecognition==3.10.0
arabic-reshaper==3.0.0
llama-index==0.12.37
llama-index-embeddings-huggingface==0.5.4
llama-index-llms-huggingface==0.5.0
python-bidi==0.6.6
sentence-transformers==2.6.1
torch==2.7.0+cu118
torchaudio==2.7.0+cu118
torchvision==0.22.0+cu118
transformers==4.52.4
huggingface_hub==0.20.3
pyaudio==0.2.13

'''
'''
errors should be considered;
1 - when individual didn't speak during recording or listening, the algorithm must stop and by pressing a click it should execute process again.
2 - embedding model should download and run locally
3 - extracting the main issue from speech like it is related to summarization and connect to appropriate LLM model.
4 - deal with inevitable errors (try-exept)
5 - stablishing a local speech to text online model
'''

# # Spech to text part libs
import speech_recognition
import os
import pyaudio
from persian_language_converter import persian_lang_converter

# # Libs related to LLM
import torch
from huggingface_hub import whoami, login
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.node_parser import SimpleNodeParser

# # these libs used for downloading LLM and embedding model
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# # Loading Embedding model from HuggingFace Hub by SentenceTransformer
emb_model = SentenceTransformer("all-MiniLM-L6-v2")
print(15*"===","\n","Your desired Embedding model loaded","\n",15*"===")

# # Download and Load desired LLM model from HuggingFace Hub
# Model_id = "name_of_model_dedicated_in_HuggingFace" # In this implementation because of system limitation we used "TinyLlama/TinyLlama-1.1B-Chat-v1.0".
# Tokenizer = AutoTokenizer.from_pretrained(Model_id)
# LLM_model = AutoModelForCausalLM.from_pretrained(Tokenizer)
# print("Your desired Pretrained LLM model downloaded and loaded")

# Checking availibity of Cuda on system
print('Cuda is available!', torch.cuda.is_available())  # Should return True
print("The version of Cuda is:",torch.version.cuda)         # Should match something like '12.1'

# Logining to API
login(token="...", add_to_git_credential=False) # Add_your_HuggingFace_Token
print(whoami())

# Loading pdf files from directory
documents = SimpleDirectoryReader("Data").load_data() 

# Parsing and Indexing all data
parser = SimpleNodeParser.from_defaults()
nodes = parser.get_nodes_from_documents(documents)

embed_model = HuggingFaceEmbedding("sentence-transformers/all-MiniLM-L6-v2")
index = VectorStoreIndex(nodes, embed_model=embed_model)

# LLM 
llm = HuggingFaceLLM(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    tokenizer_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device_map="cuda",  # or "cuda" if using GPU / "auto" if U are not sure
    max_new_tokens=64,
    context_window=512,  # or even lower
    generate_kwargs={"temperature": 0.7, "do_sample": True},
)


# Query system baised on Language model
query_engine = index.as_query_engine(llm=llm)

# os.system("cls") # It leads to clear all outputs after executing script once again.

r = speech_recognition.Recognizer()

mic_access = speech_recognition.Microphone() # This line allows us to get access to use our system microphone. 
                                             # We put it in a variable and will use it to open and close mic.

'''
Mic_names = speech_recognition.Microphone.list_microphone_names() 
Working_mic_names = speech_recognition.Microphone.list_working_microphones()

With these two script we will able to print name of microphones and also which ones already are working in our system.
'''

merge = ""

input(persian_lang_converter("برای شروع لطفا دکمه Enter را فشار دهید"))

# # Creating loop

while True:
    '''
    This loop allows us to open the microphone, denoising all noise around us, print the primary sentence, 
    # getting audio from the audience.
    '''
    with mic_access as opend_mic: # This technique (With) allows us to open the mic, apply some methods, and finally close it. opende_mic is just a simple name and has no effect.
        '''
        One of techniques for managing error is using try-except technique. 
        which means until our code is working perfectly we will be in "try". 
        If any error happened it jump into "except".
        "except" has an ability to show what error happened during executing. 
        To do this;
            except Exception as error:
                print(error)

        By this code we will able to see what happened. 
        But to manage our error and show properly to audienc, we have to return the type of error and print our sentence related to error.
        So, to do this;
            except Exception as error:
                print(type(error))
        In this task mostly we will face with two error. One of them is related to connection and the other is related to microphone
        Type error related to internet conecction is "speech_recognition.exceptions.RequestError".
        We can use that result in "except" to print our defination.
        '''
        try:   
            r.adjust_for_ambient_noise(opend_mic, duration=1) # By Recognizer from speech_recognition, we will able to apply some method on our mic, like denoising and listening to the audience.
                                                                # Here we used denoising method on our mic and set its duration on 1. 
            # with pause_threshold and non_speaking_duration we will able to manage turning off/on of mic. indeed we will manage the sensitivity of mic to voice
            r.pause_threshold = 2.5
            r.non_speaking_duration = 1.5
            
            print(persian_lang_converter("لطفا صحبت کنید")) # primary sentence

            recived_audio = r.listen(opend_mic, phrase_time_limit=30) # Listening method from Recognizer, allows us to listen and save audio of the audience.

            my_text = r.recognize_google(recived_audio, language = "fa-IR") # With this method we'll able to connect to the google and send our audio to it. 
                                                                # The important thing about this method is setting your language and accent.
                                                                # Recognizer has other APIs like Azure, Bing, etc. that allows us to connect them and use their facilities.
                                                                # To use its facilities, please refer tothe  "speech recognition" website.

            if my_text == "بای بای":
                print("=============")
                print(persian_lang_converter("به امید دیدار"))
                print("=============")
                input(persian_lang_converter("برای خروج لطفا دکمه Enter را فشار دهید")) # Without this line, whenever we want to execute the code on the terminal, it will close completely. 
                                                                                            # So, here we just prevented this issue from happening.
                break
            
            print(persian_lang_converter(f"سوال شما: {my_text}"))
            print(persian_lang_converter(" و برای بررسی به مدل زبانی ارسال گردید."))

            # response = query_engine.query(persian_lang_converter(my_text))

            os.system("cls")
            
            # print(response)
            input(persian_lang_converter("برای ادامه لطفا دکمه Enter را فشار دهید"))

            
        except speech_recognition.exceptions.RequestError:
            print(persian_lang_converter("اتصال به اینترنت خود را چک کنید"))
            input(persian_lang_converter("برای خروج لطفا دکمه Enter را فشار دهید"))
            break

        except OSError:
            print(persian_lang_converter("لطفا وضعیت میکروفن خود را چک کنید"))
            input(persian_lang_converter("برای خروج لطفا دکمه Enter را فشار دهید"))
            break
        except : # If none of those errors happened means we have another problem that we have to check it.
            print(persian_lang_converter("برنامه با مشکل مواجه شد"))
            input(persian_lang_converter("برای خروج لطفا دکمه Enter را فشار دهید"))
            break  