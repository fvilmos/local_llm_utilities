"""
This file implement utility functions to work with ollama server / package and LangChain tools i.e. RAG
Author: fvilmos
https://github.com/fvilmos

"""

from langchain_community.document_loaders import WebBaseLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from utils.prompts import PROMPT_TEMPLATE
from . import tools
from typing import get_type_hints

import ollama
import inspect
import json
import os
import cv2

def answer_a_question_msg(msg =[{"role":"user", "content":""}], 
                          model='llama2', format='',
                          tool_list=[],
                          opt= {'temperature': 0.0,'num_ctx': 4096,'seed': 42,'num_predict': 4096},
                          return_raw=False):
    """
    Usefull to answer a guestion.
    msg - list of system and user message context
    model - the desired model to be used, served by ollama
    opt - model configuration options, i.e. temperature
    """
    
    ret = ollama.chat(model=model,
                      format=format,
                      tools=tool_list,
                      options=opt,
                      messages=msg,
                     )
    if return_raw == False:
        ret = ret['message']['content']

    return ret

def answer_with_rag(question:str, source:str, model_name="llama3", options=None, top_k=2,return_raw=False, chunk_size=2800, verbose=0):
    """
    usefull for enhancing the prompt with RAG, from webpage or text
    """
    loader = None

    if verbose == 1:
        print ('loading:',source)
    
    if 'http' in source:
        print ("input type: http, ", source)
        loader = WebBaseLoader(source)

    elif '.txt' in source:
        print ("input type: txt, ", source)
        loader = TextLoader(source, autodetect_encoding=True)

    elif '.pdf' in source:
        print ("input type: pdf, ", source)
        loader = PyPDFLoader(source)
    # get the data
    try:
        data = loader.load()
    except Exception as err:
        print(f"Unexpected {err.args}")
        exit(0)

    if verbose == 1:
        print ("data lenght:", len(data))
        print ('data splitter started...')

    # split a document in smaller pieces
    splitter = TokenTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size,chunk_overlap=80)
    splits = splitter.split_documents(data)

    if verbose == 1:
        print ('data splits len:', len(splits))

    # create embeddings using ollama
    oembed = OllamaEmbeddings(model="nomic-embed-text")

    if verbose == 1:
        print ('create embeddings and feech vectors to Chroma, this may take time...')
    
    # use Chroma as vector store
    vector_db_store = Chroma.from_documents(documents=splits, embedding=oembed)

    if verbose == 1:
        print ('search for similarities in db...')
    
    # get the most relevant part from the document, using a similarity search
    docs = vector_db_store.max_marginal_relevance_search(question,k=top_k)

    # print debug information
    if verbose == 1:
        print("\n ***search results***")
        print("source: ", source)
        print(docs)

    # prepare the prompt for injection
    res_str = "\n----\n".join([doc.page_content.strip().replace("\n","").replace("\\n","").replace("\t","") for doc in docs])

    promt_formatted = PROMPT_TEMPLATE.format(context=res_str,question=question)

    # call ollama querry
    if options is not None:
        answer = answer_a_question_msg(msg=[{'role':'user', 'content':f'{promt_formatted}'}], model=model_name, opt=options, return_raw=return_raw)
    else:
        answer = answer_a_question_msg(msg=[{'role':'user', 'content':f'{promt_formatted}'}], model=model_name, return_raw=return_raw)

    return answer

def answer_with_rag_db(question, db_path:str, model_name="llama3", options=None, top_k=2,return_raw=False, verbose=0):
    """
    usefull for enhancing the prompt with data from local database with RAG
    """

    if os.path.exists(db_path) == False:
        print(f"Vector DB does not exists, check database path: {db_path}!")
        exit(0)
    
    # create embeddings using ollama
    oembed = OllamaEmbeddings(model="nomic-embed-text")

    # prepare chroma
    vector_db_store = Chroma(persist_directory=db_path, embedding_function=oembed)
    

    if verbose == 1:
        print ('search for similarities in db...')
    
    # get the most relevant part from the document, using a similarity search
    docs = vector_db_store.max_marginal_relevance_search(question,k=top_k)

    # print debug information
    if verbose == 1:
        print("\n ***search results***")
        print(docs)

    # prepare the prompt for injection
    res_str = "\n----\n".join([doc.page_content.strip().replace("\n"," ").replace("\\n"," ").replace("\t"," ") for doc in docs])
    
    promt_formatted = PROMPT_TEMPLATE.format(context=res_str,question=question)

    # print debug information
    if verbose == 1:
        print("\n ***augmented prompt***")
        print(promt_formatted)

    # call ollama querry
    if options is not None:
        answer = answer_a_question_msg(msg=[{'role':'user', 'content':f'{promt_formatted}'}], model=model_name, opt=options, return_raw=return_raw)
    else:
        answer = answer_a_question_msg(msg=[{'role':'user', 'content':f'{promt_formatted}'}], model=model_name, return_raw=return_raw)

    return answer

def answer_on_image_content(msg =[{"role":"user", "content":"", "images":[]}], 
                          model='llava', format='',
                          tool_list=[],
                          opt= {'temperature': 0.0,'num_ctx': 4096,'seed': 42,'num_predict': 4096},
                          return_raw=False, use_encoded=False):
    """
    Taks an image as input and descibs it.
    Input can be:
    - a string list
    - a source i.e. 0 = webcam
    - encoded string list, when return_raw - flag is activated

    Args:
        msg (list, optional): the prompt. Defaults to [{"role":"user", "content":"", "images":[]}].
        model (str, optional): the model used for image to text generation. Defaults to 'llava'.
        format (str, optional): '' or json. Defaults to ''.
        tool_list (list, optional): tools to be used in ollama format. Defaults to [].
        opt (dict, optional): model options. Defaults to {'temperature': 0.0,'num_ctx': 4096,'seed': 42,'num_predict': 4096}.
        return_raw (bool, optional): if to reun the model putput directly. Defaults to False.
        use_encoded (bool, optional): if input is an encoded string or a string path. Defaults to False.

    Returns:
        result: depending on the return_raw parameter can be a text of a dictionarry
    """

    # use encodeing if needed
    if use_encoded == False:
        # load files
        img_list = []
        source_list=[]

        # get the source list
        for el in msg:
            if 'images' in el:
                source_list = el['images']
                break

        # generate the encoded image list
        for fimg in source_list:
            try:
                if '.' in fimg:
                    img = cv2.imread(fimg)
                else:
                    #use another source
                    img = get_cam_image(int(fimg))
            except BaseException as err:
                print (f"Error with: {fimg}, {err}")
                continue

            en_img = cv2.imencode('.jpg', img)[1].tobytes()
            img_list.append(en_img)

        
        # check if input available
        if len(img_list)==0:
            print ("No input! Exiting")
            exit()
        
        #update message with encoded image list
        for el in msg:
            if 'images' in el:
                el['images'] = img_list
                break
    else:
        # use directly the input
        pass

    ret = ollama.chat(model=model,
                    options=opt,
                    messages=msg,
                    format=format,
                    tools=tool_list
                    )
    
    if return_raw == False:
        ret = ret['message']['content']

    return ret

def get_cam_image(source:int=0, skipp_frames=15, delay:int=10):
    """
    Get a picture from a source specified

    Args:
        source (int, optional): By defaul use webcam. Defaults to 0.
        skipp_frames (int, optional): Number of frames skipped before image returned. Defaults to 15.

    Returns:
        array: image
    """
    cam = cv2.VideoCapture(source)
    rimg = None
    _,img = cam.read()
    
    count = skipp_frames
    get_images = True

    while get_images:
        _,img = cam.read()
        #cv2.waitKey(delay)
        if (img is None):
            continue
        if count >=0: 
            count -=1
        else:
            get_images = False
    del(cam)

    # test if image exist
    if img is not None:
        # show image
        rimg = img
    else:
        print ("something went wrong, check camera source!")
    return rimg


def get_tools():
    """
    return a set with the availble tool names
    """
    item_list = dir(tools)
    fn_names = {f"{it}" for it in item_list if it[0]!="_"}
    return fn_names

def function2json(fn):
    """
    Transforms a function to ollma function call structure.
    https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-completion

    i.e. def get_weather(city: str="") ==>
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "current weather for a location",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {
              "type": "string",
              "description": "city"
            },
          },
          "required": ["city"]
        }
      }
    }
    """
    # get function signature
    signature = inspect.signature(fn)
    type_hints = get_type_hints(fn)

    # create the function structure, ollama style
    function_structure = {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": str(fn.__doc__).strip().replace("\n","").replace("\\","").replace("\t",""),
            "parameters": {
                "type":"object",
                "properties":{},
            },
        },
        #"return": type_hints.get("return", "void").__name__ if str(type_hints.get("return", "void")) != "void" else None,
    }
    
    # fill function parameters
    for name, _ in signature.parameters.items():
        
        # get type of the function parameter
        p_type = type_hints.get(name, type(None))
        p_type_filtered = p_type.__name__ if 'class' in str(p_type) else p_type

        # update function dict
        function_structure["function"]["parameters"]["properties"][name] = {"type": p_type_filtered}
    p_temp  =  function_structure["function"]["parameters"]["properties"]
    
    p_param = function_structure["function"]["parameters"]
    p_param["required"]=list(dict(p_temp).keys())

    return json.dumps(function_structure, indent=2)